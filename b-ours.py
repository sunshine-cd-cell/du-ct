import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
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
from torchvision.utils import make_grid
import torch.nn as nn
from networks.hhhh import VNet
from networks.unetr import UNETR
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler,RandomCrop1
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#import wandb
import datetime

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/root/brats_data/pro/train', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='Kours_10', help='model_name')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=20, help='consistency_rampup')
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
dice_loss = losses.DiceLoss(2)
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
def calculate_consistency_loss(preds, outputs_aux_soft_list, kl_distance, args):
    consistency_loss_list = []
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

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


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
       # dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def compute_unsupervised_loss(predict, target, percent, pred_teacher,pp,uncer):
    # batch_size, num_class, h, w = predict.shape
    # Focal_Loss=FocalLoss(alpha=[1,100], gamma=2, num_classes=2, size_average=True)
    with torch.no_grad():
        #entropy = -torch.sum(pred_teacher * torch.log(pred_teacher + 1e-10), dim=1)

        pp = F.softmax(pp)
        entropy = uncer
        thresh = 0
        logits_u_aug_start, label_u_aug_start = torch.max(pp, dim=1)
        # 1

        mask = label_u_aug_start == 1
        entropy1 = entropy[mask]


        if entropy1.numel() == 0:
            thresh1 = 0
            mean_entropy1=0
        else:
            thresh1 = np.percentile(
                entropy1.detach().cpu().numpy().flatten(), percent
            )
            filtered_entropy1 = entropy1[entropy1 <= thresh1]



            if filtered_entropy1.numel() == 0:
                entropy1_cpu = entropy1.cpu()

                # 将其转换为 NumPy 数组
                data = entropy1_cpu.numpy()
                entropy11_cpu = filtered_entropy1.cpu()

                # 将其转换为 NumPy 数组
                data1 = entropy11_cpu.numpy()
                mean_entropy1 = 0
                print('oo',len(data),len(data1))
                print(data)
           # 计算筛选后元素的平均值
            else:
                mean_entropy1 = filtered_entropy1.mean().item()
        # 0
        mask = label_u_aug_start == 0

        entropy0 = entropy[mask]
        # min_entropy0 = torch.min(entropy0)

        mean_entropy0 = torch.mean(entropy0)

    predict_soft = torch.softmax(predict, dim=1)
    predict_soft = predict_soft[:, 1, :, :, :]

    loss_seg_dice = losses.dice_loss(predict_soft, target == 1)

    # loss = F.cross_entropy(predict, target, ignore_index=255)  # [10, 321, 321]
    # loss = Focal_Loss(predict, target)  # [10, 321, 321]
    # print (loss)

    hebing = -(torch.tensor(mean_entropy0) + torch.tensor(mean_entropy1))
    uncertainty_loss =  loss_seg_dice*torch.exp(hebing)
    # uncertainty_loss = (torch.exp(-entropy) * loss_seg_dice).mean()
    print(torch.exp(hebing))
    return uncertainty_loss, thresh
    # return loss,thresh

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
  for k in range(1,5):
    x=f'{k}train.txt'
    print(x)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
   # if os.path.exists(snapshot_path + '/code'):
   #     shutil.rmtree(snapshot_path + '/code')
   # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/10ours_zishiyinglog.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))





    model = VNet(n_channels=1, n_classes=num_classes, normalization='instancenorm', has_dropout=True).cuda()
    '''
    checkpoint = torch.load('/root/brats/model/ours/v5000iter_4000.pth')
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Ignoring size mismatch errors while loading state_dict: {e}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters loaded into the model: {num_params}")
    '''


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
    '''
    checkpoint = torch.load(
        '/root/brats/model/ours/t5000iter_4000.pth')  # 假设预训练权重保存在 pretrained_model.pth 文件中
    # trans.load_state_dict(checkpoint['state_dict'],strict=False)
    try:
        trans.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"gggggggggggggggIgnoring size mismatch errors while loading state_dict: {e}")

    num_params = sum(p.numel() for p in trans.parameters() if p.requires_grad)
    print(f"Number of parameters loaded into the model: {num_params}")
    '''
  #  wandb.init(
        # set the wandb project where this run will be logged
     #   project="xiaorong", name='brats_20_wuyu')
    train_log = {}
    val_log = {}
    train_log1={}
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       train_fold=x,
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop((96, 96, 96)),
                           ToTensor(),
                       ]))
    db_test = LAHeart(base_dir=train_data_path,
                      split='test',
                      train_fold=x,
                      transform=transforms.Compose([
                           CenterCrop((96, 96, 96)),
                           ToTensor()
                      ]))
    labeled_idxs = list(range(ul))
    unlabeled_idxs = list(range(ul, 290))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_test, batch_size=1, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()

    trans.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer1 = optim.SGD(trans.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=4000,
        eta_min=0.0001,
        last_epoch=-1
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1,
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
    dice1 = 0
    dice2= 0
    iter_num =0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    average_dices = []
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        model.train()

        trans.train()
        for i_batch, sampled_batch in enumerate(trainloader):

            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            consistency_weight = get_current_consistency_weight(
                iter_num // 150)

            unlabeled_volume_batch = volume_batch[labeled_bs:]


            outputs,v2,v1,v0 = model(volume_batch)
            outputs_aux1_softv = torch.softmax(outputs, dim=1)
            outputs_aux2_softv = torch.softmax(v2, dim=1)
            outputs_aux3_softv = torch.softmax(v1, dim=1)
            outputs_aux4_softv = torch.softmax(v0, dim=1)
            predsv = (outputs_aux1_softv +
                     outputs_aux2_softv + outputs_aux3_softv + outputs_aux4_softv) / 4
            outputs_aux_softv_list = [outputs_aux1_softv, outputs_aux2_softv, outputs_aux3_softv, outputs_aux4_softv]
            outputsv_list = [outputs, v2, v1, v0]

            #trans_outputs = trans(volume_batch).cuda()
            logits, dec2, dec1, dec0 = trans(volume_batch)

            outputs_aux1_soft = torch.softmax(logits, dim=1)
            outputs_aux2_soft = torch.softmax(dec2, dim=1)
            outputs_aux3_soft = torch.softmax(dec1, dim=1)
            outputs_aux4_soft = torch.softmax(dec0, dim=1)
            preds = (outputs_aux1_soft +
                     outputs_aux2_soft + outputs_aux3_soft + outputs_aux4_soft) / 4
            outputs_aux_soft_list = [outputs_aux1_soft, outputs_aux2_soft, outputs_aux3_soft, outputs_aux4_soft]
            outputst_list = [logits, dec2, dec1, dec0]

            ###
            ## calculate the loss

            supervised_loss1 = calculate_supervised_loss(outputst_list, label_batch, args)

            supervised_loss = calculate_supervised_loss(outputsv_list, label_batch, args)
            consistency_dist = torch.mean((preds[args.labeled_bs:] - predsv[args.labeled_bs:]) ** 2)
            if iter_num>0:
              uncert = calculate_consistency_loss(preds, outputs_aux_soft_list, kl_distance, args)
              uncerv = calculate_consistency_loss(predsv, outputs_aux_softv_list, kl_distance, args)
              drop = 100
              drop1 = 100

              pred_u_teacher = F.softmax(outputs[labeled_bs:], dim=1)
              logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)
              consistency_loss, thresh = compute_unsupervised_loss(logits[labeled_bs:], label_u_aug.clone(), drop1,
                                                                     predsv[labeled_bs:].detach(),
                                                                     outputs[labeled_bs:].detach(), uncerv)

              combine = logits[labeled_bs:]
              pred_u_z = F.softmax(combine, dim=1)
              logits_u_aug1, label_u_aug1 = torch.max(pred_u_z, dim=1)
              consistency_loss_s_z, thresh = compute_unsupervised_loss(outputs[labeled_bs:], label_u_aug1.clone(),
                                                                         drop,
                                                                         preds[labeled_bs:].detach(), combine.detach(),
                                                                         uncert)

              outputs_softt = F.softmax(logits, dim=1)


              outputs_soft = F.softmax(outputs, dim=1)


              logits_u_augt, label_u_augt = torch.max(outputs_softt[:labeled_bs], dim=1)
              logits_u_augv, label_u_augv = torch.max(outputs_soft[:labeled_bs], dim=1)
              dicet = cal_dice(label_u_augt.cpu().numpy(), label_batch[:labeled_bs].cpu().numpy())
              dicev = cal_dice(label_u_augv.cpu().numpy(), label_batch[:labeled_bs].cpu().numpy())
              epsilon = 1e-8
              meand = (dicet + dicev) / 2

              ratio_v = dicev / (dicet + epsilon)
              ratio_t = 1 / (ratio_v + epsilon)

              tensort = torch.from_numpy(dicet)
              tensorv = torch.from_numpy(dicev)
              tensorm = torch.from_numpy(meand)
              if dicev > dicet:
                  coeff_t = math.cos(math.pi * tanh(3 * (tensorv - tensort)) / 2)
                  coeff_v = 1
                  coeff_v = coeff_v
                  # coeff_t = torch.from_numpy(coeff_t).cuda()
              else:
                  coeff_v = math.cos(math.pi * tanh(3 * (tensort - tensorv)) / 2)
                  coeff_t = 1
                  # coeff_v = torch.from_numpy(coeff_v).cuda()
                  coeff_t = coeff_t
              print(dicev, dicet, coeff_v, coeff_t)

            # consistency_weight = get_current_consistency_weight(iter_num//150)
            # consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output) #(batch, 2, 112,112,80)
            # loss =consistency_weight* consistency_loss + consistency_weight*consistency_loss_s_z
            # print(loss)
            # loss = supervised_loss + supervised_loss1
              loss = supervised_loss + supervised_loss1 +coeff_v*consistency_weight * consistency_loss + coeff_t*consistency_weight * consistency_loss_s_z+ consistency_weight+consistency_dist
            # loss =  consistency_loss +  consistency_loss_s_z
              optimizer.zero_grad()
              optimizer1.zero_grad()
              loss.backward()


              optimizer.step()
              optimizer1.step()
              logging.info('iteration %d : loss : %f superv:%f,supert:%f,cons_loss: %f, cons_loss_s_z: %f ' %
                           (iter_num, loss.item(), supervised_loss.item(), supervised_loss1.item(),
                            consistency_loss.item(), consistency_loss_s_z.item()))

            else:
              optimizer.zero_grad()
              optimizer1.zero_grad()


              loss=supervised_loss+supervised_loss1
              loss.backward()

              optimizer.step()
              optimizer1.step()
              print('iteration %d : loss : %f superv:%f,supert:%f' %
                           (iter_num, loss.item(), supervised_loss.item(), supervised_loss1.item()
                            ))
         #   wandb.log({"train loss": loss.item(), 'iteration': iter_num})
         #   wandb.log({"train/superv": supervised_loss.item(), 'iteration': iter_num})
          #  wandb.log({"train/supert": supervised_loss1.item(), 'iteration': iter_num})
            iter_num = iter_num + 1
            scheduler.step()
            scheduler1.step()
            if iter_num>8000 and iter_num % \
                    1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' +str(k)+'_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if iter_num >= max_iterations:
                break
            time1 = time.time()

            if iter_num > 16000 and iter_num % 28 == 0:

                model.eval()
                total_dice = 0
                # 假设每个 batch 包含的样本数量为 batch_size
                total_samples = 0
                for i, data in enumerate(valloader):
                    with torch.no_grad():
                        volume_batch, label_batch = data['image'], data['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                        # volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                        outputs = model(volume_batch)

                        logits_u_aug12, label_u_aug12 = torch.max(outputs[0], dim=1)
                        dice = cal_dice(label_u_aug12.cpu().numpy(), label_batch.cpu().numpy())
                        total_dice += dice
                        # 假设每个 batch 包含的样本数量为 batch_size
                        total_samples = i + 1
                average_dice = total_dice / total_samples
                average_dices.append(average_dice)
                print('average_dice:' + str(average_dice))
                if average_dice > dice1:
                    save_mode_path = os.path.join(snapshot_path, str(k) + '_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("Save best model to {}".format(save_mode_path))
                    dice1 = average_dice
                print(dice1)
                model.train()
        if iter_num >= max_iterations:
            break
            ###val