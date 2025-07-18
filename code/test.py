import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/root/brats_data/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='ours', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(os.path.join( '/root/brats/data/test.list.txt'), 'r') as f:
    image_list = f.readlines()
image_list = ['/root/brats_data/pro/train/' +item.replace('\n', '')+".h5" for item in image_list]


def test_calculate_metric(epoch_num):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm', has_dropout=True).cuda()#ssas canet dtc
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path),strict=False)

    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric,sen = test_calculate_metric(20000)
    print(metric,sen)


