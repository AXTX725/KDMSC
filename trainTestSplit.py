import numpy as np
from scipy.io import loadmat
import os
import random
from scipy import io

# Dictionary mapping dataset names to their corresponding ground truth keys
keys = {'PaviaU':'paviaU_gt',
        'Salinas':'salinas_gt',
        'indian_pines_corrected':'indian_pines_gt',
        'Houston':'Houston_gt',
        'KSC':'KSC_gt',
        'gf5': 'gf5_gt',
        'Xiongan': 'xiongan_gt'}
TRAIN_SIZE = [25]   # Number of labeled samples during training
RUN = 10      # # total number of partitioned data files


def sample_gt(gt, train_size, mode='fixed_withone'):
    '''
    Sample training and testing data from the full ground truth map.

    Parameters:
    - gt: ground truth label map
    - train_size: number of samples per class or percentage (if < 1)
    - mode: sampling strategy, supports 'fixed_withone' and 'random_withone'

    Returns:
    - train_gt: training label map
    - test_gt: testing label map
    '''
    indices = np.nonzero(gt)
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)
        if mode == 'random':
            train_size = float(train_size) / 100

    if mode == 'random_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            train_len = int(np.ceil(train_size * len(X)))
            train_indices += random.sample(X, train_len)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0

    elif mode == 'fixed_withone':
        train_indices = []
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features

            train_indices += random.sample(X, train_size)
        index = tuple(zip(*train_indices))
        train_gt[index] = gt[index]
        test_gt[index] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


# 保存样本
def save_sample(train_gt, test_gt, dataset_name, sample_size, run):
    '''
      Save the train/test label maps into .mat format.

      Parameters:
      - train_gt: training label map
      - test_gt: testing label map
      - dataset_name: name of the dataset
      - sample_size: number of training samples per class
      - run: current run index
      '''
    sample_dir = './trainTestSplit/' + dataset_name + '/'
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    sample_file = sample_dir + 'sample' + str(sample_size) + '_run' + str(run) + '.mat'
    io.savemat(sample_file, {'train_gt':train_gt, 'test_gt':test_gt})


def load(dname):
    '''
    Load the ground truth label map from .mat file.

    Parameters:
    - dname: dataset name

    Returns:
    - gt: ground truth label map
    '''
    path = os.path.join(dname,'{}_gt.mat'.format(dname))
    dataset = loadmat(path)
    key = keys[dname]
    gt = dataset[key]
    # # 采样背景像素点
    # gt += 1
    return gt


def TrainTestSplit(datasetName):
    '''
    Perform repeated train/test splitting for a given dataset.

    Parameters:
    - datasetName: name of the dataset
    '''
    gt = load(datasetName)
    for size in TRAIN_SIZE:
        for r in range(RUN):
            train_gt, test_gt = sample_gt(gt, size)
            save_sample(train_gt, test_gt, datasetName, size, r)
    print('Finish split {}'.format(datasetName))


if __name__ == '__main__':
    dataseteName = ['indian_pines_corrected']
    for name in dataseteName:
        TrainTestSplit(name)
    print('*'*8 + 'FINISH' + '*'*8)