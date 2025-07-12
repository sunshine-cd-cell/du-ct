import h5py
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# 四种模态的mri图像


# train
train_set = {
    'root': '' ,
    'out': '',
    'flist': '',
    'nii_out': ''

}

def get_none_zero_region(im, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = im.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(im)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output

def process_h5(path, out_path,nii_out):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    # SimpleITK读取图像默认是是 DxHxD，这里转为 HxWxD
    margin = 5
    label = sitk.GetArrayFromImage(sitk.ReadImage(path + 'seg.nii')).transpose(1, 2, 0)
    print(label.shape)
    label[label != 0] = 1
    print('Unique values in label:', np.unique(label))
    # 堆叠四种模态的图像，4 x (H,W,D) -> (4,H,W,D)
    images = sitk.GetArrayFromImage(sitk.ReadImage(path + 'flair' + '.nii')).transpose(1, 2, 0)
         # [240,240,155]

    # 数据类型转换
    label = label.astype(np.uint8)
    if label.size == 0:
        print("label 是一个空数组，没有标签数据。")
    images = images.astype(np.float32)

    bbmin, bbmax = get_none_zero_region(images, margin)
    images = crop_ND_volume_with_bounding_box(images, bbmin, bbmax)
    label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax)
    if label.size == 0:
        print("croplabel 是一个空数组，没有标签数据。")
    print(images.shape,label.shape)
    case_name = path.split('/')[-1]
    # case_name = os.path.split(path)[-1]  # windows路径与linux不同
    nii_yuan_out='/root/brats_data/pro/nii_yuan/'
    save_as_nii(images, label, nii_yuan_out, case_name)

    path = os.path.join(out_path, case_name)
    output = path + 'mri_norm2.h5'

    mask = images.sum(0) > 0
    print(np.min(images))

    pixels = images[images > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (images - mean) / std

    out = out.astype(np.float32)
    print(mean,std,np.max(out),np.min(out))
    print(case_name, images.shape, label.shape)
    f = h5py.File(output, 'w')
    f.create_dataset('image', data=out, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()
    save_as_nii(images, label, nii_out, case_name)
def save_as_nii(images, label, nii_out, case_name):
    """ Save the images and label as .nii files. """
    if not os.path.exists(nii_out):
        os.makedirs(nii_out)

    image_sitk = sitk.GetImageFromArray(images.transpose(2, 0, 1))
    label_sitk = sitk.GetImageFromArray(label.transpose(2, 0, 1))

    sitk.WriteImage(image_sitk, os.path.join(nii_out, case_name + '_image.nii'))
    sitk.WriteImage(label_sitk, os.path.join(nii_out, case_name + '_label.nii'))
def doit(dset):
    root, out_path,nii_out  = dset['root'], dset['out'],dset['nii_out']
    paths = [os.path.join(root, name, name + '_') for name in os.listdir(root)]

    for path in tqdm(paths):
        process_h5(path, out_path,nii_out)
        # break
    print('Finished')


if __name__ == '__main__':
    doit(train_set)
