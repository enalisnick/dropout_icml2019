from random import shuffle
import _pickle as cp
import numpy as np
from sklearn.preprocessing import StandardScaler


def convert_yacht():
    data = []

    with open("yacht_data/raw/yacht.txt") as f:
        for line in f.readlines():
            data.append([float(x) for x in line.strip().split()])

    data = np.array(data)
    return data[:,:6], data[:,6][np.newaxis].T



def main():

    dataset_names = ["yacht"] 
    n_splits = 20

    for dataset_name in dataset_names:

        all_x_data_train = []
        all_y_data_train = []
        all_x_data_test = []
        all_y_data_test = []
        x_scalers = []
        y_scalers = []
        for split_idx in range(n_splits):

            x, y = convert_yacht()
            assert x.shape[0] == y.shape[0]

            idxs = list(range(x.shape[0]))
            shuffle(idxs)
            cutoff_idx = int(.9 * len(idxs))
            train_idxs = idxs[:cutoff_idx]
            test_idxs = idxs[cutoff_idx:]

            x_train = x[train_idxs,:] 
            y_train = y[train_idxs,:]
            x_test = x[test_idxs,:]
            y_test = y[test_idxs,:]

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
        
            x_train = x_scaler.fit_transform(x_train)
            y_train = y_scaler.fit_transform(y_train)
            x_test = x_scaler.transform(x_test)
            
            x_scalers.append(x_scaler)
            y_scalers.append(y_scaler)

            all_x_data_train.append(x_train)
            all_y_data_train.append(y_train)
            all_x_data_test.append(x_test)
            all_y_data_test.append(y_test)

        cp.dump([np.array(all_x_data_train), np.array(all_x_data_test), np.array(all_y_data_train), np.array(all_y_data_test), x_scalers, y_scalers], open("yacht_data/pkl/%s.pkl"%(dataset_name), "wb"))


if __name__ == "__main__":
    main()
