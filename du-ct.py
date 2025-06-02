import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
import time
import random
import numpy as np
from monai.networks.nets import SwinUNETR
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler,RandomCrop1
import matplotlib
import math
matplotlib.use('Agg')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/root/brats_data/pro/train', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='5_K_ours', help='model_name')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=130, help='consistency_rampup')
args = parser.parse_args()
ul=29
train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"
kl_distance = nn.KLDivLoss(reduction='none')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
tanh = nn.Tanh()
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
num_classes = 2
patch_size = (96, 96, 96)
relu = nn.ReLU(inplace=True)

def calculate_supervised_loss(outputs_list, label_batch, args):
    # 计算交叉熵损失和Dice损失
    loss_ce_list = []
    loss_dice_list = []
    for i in range(len(outputs_list)):
        # 计算交叉熵损失
        loss_ce = F.cross_entropy(outputs_list[i][:args.labeled_bs], label_batch[:args.labeled_bs])
        loss_ce_list.append(loss_ce)
        outputs_soft = F.softmax(outputs_list[i], dim=1)
        # 计算Dice损失
        loss_dice = losses.dice_loss(outputs_soft[:labeled_bs,1,:,:,:],
                                          label_batch[:labeled_bs]==1)
        loss_dice_list.append(loss_dice)
    # 计算总的监督损失
    supervised_loss = (sum(loss_ce_list) + sum(loss_dice_list)) / (2 * len(outputs_list))
    return supervised_loss

def calculate_uncer(preds, outputs_aux_soft_list, kl_distance, args):
    variance_aux_list = 0
    for outputs_aux_soft in outputs_aux_soft_list:
        variance_aux = torch.sum(kl_distance(torch.log(outputs_aux_soft[args.labeled_bs:]), preds[args.labeled_bs:]),
                                 dim=1, keepdim=True)
        variance_aux_list=variance_aux_list+variance_aux
    # 计算平均值
    mean_variance_aux = variance_aux_list/4
    return mean_variance_aux.squeeze(1)

class CustomSwinUNETR(SwinUNETR):
    def __init__(self, **kwargs):
        super(CustomSwinUNETR, self).__init__(**kwargs)
        self.dsv2 = UnetDsv3(in_size=48, out_size=2, scale_factor=8).cuda()
        self.dsv1 = UnetDsv3(in_size=24, out_size=2, scale_factor=4).cuda()
        self.dsv0 = UnetDsv3(in_size=12, out_size=2, scale_factor=2).cuda()
        self.dsv3 = nn.Conv3d(in_channels=12, out_channels=2, kernel_size=1).cuda()
        self.out = None
    def forward(self, x_in):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])

        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])

        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)

        out = self.decoder1(dec0, enc0)
        out = self.dsv3(out)
        dec2 = self.dsv2(dec2)
        dec1 = self.dsv1(dec1)
        dec0 = self.dsv0(dec0)

        return  out,dec2, dec1, dec0

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float32)
        label_tmp = label_tmp.astype(np.float32)
        intersection = np.sum(prediction_tmp * label_tmp)
        denominator = np.sum(prediction_tmp) + np.sum(label_tmp)
        if denominator == 0:
            dice = 0  # 如果你觉得“标签和预测都没这个类”算完全一致
        else:
            dice = 2 * intersection / denominator
        total_dice[i - 1] += dice
    return total_dice

def compute_unsupervised_loss(predict, target,pp,uncer):
    with torch.no_grad():
        pp = F.softmax(pp)
        logits_u_aug_start, label = torch.max(pp.detach(), dim=1)
        # 1
        mask = label == 1
        entropy1 = uncer[mask]
        if entropy1.numel() == 0:
            mean_entropy1=0
        else:
           # 计算筛选后元素的平均值
            mean_entropy1 = entropy1.mean().item()
        # 0
        mask = label == 0
        entropy0 = uncer[mask]
        mean_entropy0 = torch.mean(entropy0)

    predict_soft = torch.softmax(predict, dim=1)
    predict_soft = predict_soft[:, 1, :, :, :]
    loss_seg_dice = losses.dice_loss(predict_soft, target == 1)
    hebing = -(torch.tensor(mean_entropy0) + torch.tensor(mean_entropy1))
    usue_cpsloss = loss_seg_dice * torch.exp(hebing)
    print(torch.exp(hebing))
    return usue_cpsloss
class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )
    def forward(self, input):
        return self.dsv(input)
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
if __name__ == "__main__":
    ## make logger file
  for k in range(0,5):
    x=f'{k}train.txt'
    print(x)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/10ours_zishiyinglog.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    model = VNet(n_channels=1, n_classes=num_classes, normalization='instancenorm', has_dropout=True).cuda()
    trans = CustomSwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=2,
        feature_size=12,
        drop_rate=0.2,
        attn_drop_rate=0.2,
        dropout_path_rate=0.2,
        use_checkpoint=True,
    ).cuda()
    db_train = Brats(base_dir=train_data_path,
                       split='train',
                       train_fold=x,
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop((96, 96, 96)),
                           ToTensor(),
                       ]))

    labeled_idxs = list(range(ul))
    unlabeled_idxs = list(range(ul, 290))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer1 = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(trans.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1,
        T_max=4000,
        eta_min=0.0001,
        last_epoch=-1
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2,
        T_max=4000,
        eta_min=0.0001,
        last_epoch=-1
    )
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num =0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        model.train()
        trans.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            #cnn
            outputs,v2,v1,v0 = model(volume_batch)
            v_outputs_aux1_soft = torch.softmax(outputs, dim=1)
            v_outputs_aux2_soft = torch.softmax(v2, dim=1)
            v_outputs_aux3_soft = torch.softmax(v1, dim=1)
            v_outputs_aux4_soft = torch.softmax(v0, dim=1)
            mean_preds_v = (v_outputs_aux1_soft +
                     v_outputs_aux2_soft + v_outputs_aux3_soft + v_outputs_aux4_soft) / 4
            v_outputs_aux_soft_list = [v_outputs_aux1_soft, v_outputs_aux2_soft, v_outputs_aux3_soft, v_outputs_aux4_soft]
            v_outputs_list = [outputs, v2, v1, v0]
            #transformer
            logits,t2, t1, t0 = trans(volume_batch)
            t_outputs_aux1_soft = torch.softmax(logits, dim=1)
            t_outputs_aux2_soft = torch.softmax(t2, dim=1)
            t_outputs_aux3_soft = torch.softmax(t1, dim=1)
            t_outputs_aux4_soft = torch.softmax(t0, dim=1)
            mean_preds_t = (t_outputs_aux1_soft +
                     t_outputs_aux2_soft + t_outputs_aux3_soft + t_outputs_aux4_soft) / 4
            t_outputs_aux_soft_list = [t_outputs_aux1_soft, t_outputs_aux2_soft, t_outputs_aux3_soft, t_outputs_aux4_soft]
            outputst_list = [logits, t2, t1, t0]
            
            
            ###
            ## calculate the loss
            supervised_loss_t = calculate_supervised_loss(outputst_list, label_batch, args)
            supervised_loss_v = calculate_supervised_loss(v_outputs_list, label_batch, args)
            ###consitency
            consistency_dist = torch.mean((mean_preds_t[args.labeled_bs:] - mean_preds_v[args.labeled_bs:]) ** 2)
            #cpsloss
            uncer_t = calculate_uncer(mean_preds_t, t_outputs_aux_soft_list, kl_distance, args)
            uncer_v = calculate_uncer(mean_preds_v, v_outputs_aux_soft_list, kl_distance, args)
            v_u_preds = F.softmax(outputs[labeled_bs:], dim=1)
            v_u_logits, v_u_label = torch.max(v_u_preds.detach() , dim=1)
            cpsloss_v= compute_unsupervised_loss(logits[labeled_bs:], v_u_label.clone(),                                                   
                                                                     outputs[labeled_bs:].detach(), uncer_v)
            t_u_preds = F.softmax(logits[labeled_bs:], dim=1)
            t_u_logits, t_u_label = torch.max(t_u_preds.detach(), dim=1)
            cpsloss_t = compute_unsupervised_loss(outputs[labeled_bs:], t_u_label.clone(),                                                                   
                                                                        logits[labeled_bs:].detach(),uncer_t)
            
            t_logits_u_aug, t_label_u_aug = torch.max(t_outputs_aux1_soft[:labeled_bs].detach(), dim=1)
            v_logits_u_aug, v_label_u_aug = torch.max(v_outputs_aux1_soft[:labeled_bs].detach(), dim=1)

            t_dice = cal_dice(t_label_u_aug.cpu().numpy(), label_batch[:labeled_bs].cpu().numpy())
            v_dice = cal_dice(v_label_u_aug.cpu().numpy(), label_batch[:labeled_bs].cpu().numpy())
            
            t_dice_tensor = torch.from_numpy(t_dice)
            v_dice_tensor = torch.from_numpy(v_dice)
            
            if v_dice > t_dice:
                  coeff_t = math.cos(math.pi * tanh(3 * (v_dice_tensor - t_dice_tensor)) / 2)
                  coeff_v = 1
            else:
                  coeff_v = math.cos(math.pi * tanh(3 * (t_dice_tensor - v_dice_tensor)) / 2)
                  coeff_t = 1
            print(v_dice, t_dice, coeff_v, coeff_t)
            
            
            loss = supervised_loss_v + supervised_loss_t +coeff_v * consistency_weight * cpsloss_v + coeff_t * consistency_weight * cpsloss_t + consistency_weight*consistency_dist
            optimizer1.zero_grad()
            optimizer2.zero_grad()
        
            loss.backward()
            
            optimizer1.step()
            optimizer2.step()
            logging.info('iteration %d : loss : %f superv:%f,supert:%f ,cpsloss_v: %f, cpsloss_t: %f ' %
                           (iter_num, loss.item(), supervised_loss_v.item(), supervised_loss_t.item(),
                            cpsloss_v.item(), cpsloss_t.item()))
            iter_num = iter_num + 1
            scheduler.step()
            scheduler1.step()
            if iter_num>8000 and iter_num % \
                    200 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' +str(k)+'_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
            ###val


