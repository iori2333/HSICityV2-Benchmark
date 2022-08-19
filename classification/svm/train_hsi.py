from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader
from dataset import HSICity2
from svm.function import train_average
import pickle
import logging

def main():
    train_set = HSICity2("/data/huangyx/data/HSICityV2/train", use_hsi=True)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4)

    # svc + rbf
    svc = SVC(C=40, kernel="rbf", gamma=1.0, random_state=42, verbose=True)
    print('\n')
    # linear svc
    l_svc = LinearSVC(C=40, random_state=42, max_iter=10000, verbose=True)
    knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)

    train_average(svc, train_loader, sz=1)
    train_average(l_svc, train_loader, sz=1)
    train_average(knn, train_loader, sz=1)
    
    with open('svc_hsi.pkl', 'wb') as f:
        pickle.dump(svc, f)
    with open('linearsvc_hsi.pkl', 'wb') as f:
        pickle.dump(l_svc, f)
    with open('knn_hsi.pkl', 'wb') as f:
        pickle.dump(knn, f)
    logging.info('training done')


if __name__ == "__main__":
    logging.basicConfig(filename='hsi_log.log', format='%(asctime)-15s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    main()
