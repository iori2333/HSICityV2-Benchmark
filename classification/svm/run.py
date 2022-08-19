import argparse
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from dataset import HSICity2
from svm.function import validate
import pickle
import logging

def main(config):
    if 'hsi' in config.model:
        use_hsi = True
    else:
        use_hsi = False
    test_set = HSICity2("/data/HSICityV2/test", use_hsi=use_hsi)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    with open(config.model, 'rb') as f:
        svc = pickle.load(f)

    validate(svc, test_loader, sz=config.size, out=config.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument('--size', type=int, default=1)
    parser.add_argument('--out', type=str)
    config = parser.parse_args()
    logging.basicConfig(filename=config.log, format='%(asctime)-15s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    main(config)
