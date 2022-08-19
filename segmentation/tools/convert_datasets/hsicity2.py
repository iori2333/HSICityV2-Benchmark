import numpy as np
import argparse
import os
import os.path as osp
from tqdm import tqdm
import mmcv
import pickle


def read_hsd(filename):
    data = np.fromfile(filename, dtype=np.int32)
    height = data[0]
    width = data[1]
    SR = data[2]
    D = data[3]

    data = np.fromfile(filename, dtype=np.float32)
    a = 7
    average = data[a:a + SR]
    a = a + SR
    coeff = data[a:a + D * SR].reshape((D, SR))
    a = a + D * SR
    scoredata = data[a:a + height * width * D].reshape((height * width, D))

    temp = np.dot(scoredata, coeff)
    data = (temp + average).reshape((height, width, SR))

    return data


def convert_hsd(filename):
    data = read_hsd(filename)
    data = mmcv.imrescale(data, 0.5, interpolation='nearest')
    return data


def parse_args():
    parser = argparse.ArgumentParser(description='convert hsd to hsi')
    parser.add_argument('--root', type=str, help='hsicity2 root')
    args = parser.parse_args()
    return args


def main():
    config = parse_args()
    train_dir = osp.join(config.root, 'train')
    test_dir = osp.join(config.root, 'test')

    for dir in [train_dir, test_dir]:
        print('converting {}'.format(dir))
        hsds = filter(lambda x: x.endswith('.hsd'), os.listdir(dir))
        for filename in tqdm(hsds):
            if filename.endswith('.hsd'):
                data = convert_hsd(osp.join(dir, filename))
                path = osp.join(dir, filename[:-4] + '.pkl')
                with open(path, 'wb') as f:
                    pickle.dump(data, f)

if __name__ == '__main__':
    main()
