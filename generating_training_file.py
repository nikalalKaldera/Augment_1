import numpy as np
import argparse
import pickle
#from demo_object import *

def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 60))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector

parser = argparse.ArgumentParser(description='Frequency analysise')
parser.add_argument('-o', '--original', default='/home/ntkkh958/My_work/Codes/CTR_NEW_26/CTR-GCN/data/ntu120/NTU120_CSub.npz',
                    help='location of dataset npz file')
parser.add_argument('-a', '--augment', default='./Augmented_sequence.pkl',
                    help='location of dataset npz file')
parser.add_argument('-l', '--auglabel', default='./aug_label.pkl',
                    help='location of dataset npz file')


arg = parser.parse_args()  # Parse the command-line arguments.
print('Data file read start..')
npz_data = np.load(arg.original)
data_orig_train, label_orig_train = npz_data['x_train'], np.where(npz_data['y_train'] > 0)[1]
# x_test, y_test = npz_data['x_test'], np.where(npz_data['y_test'] > 0)[1]
#test_x, test_y = npz_data['x_test'], npz_data['y_test']
del npz_data
N, T, _ = data_orig_train.shape
data_orig_train_1 = data_orig_train.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N C T V M
data_orig_train_2 = data_orig_train_1.transpose(0, 1, 2, 3, 4)    # N M T V C
del data_orig_train, data_orig_train_1

with open(arg.augment, 'rb') as file:
    aug_data = np.array(pickle.load(file))
with open(arg.auglabel, 'rb') as file1:
    aug_label = np.array(pickle.load(file1))

# Create a new array of zeros with the desired shape (80, 3, 300, 25, 2)
padded_aug = np.zeros((aug_data.shape[0], aug_data.shape[1], data_orig_train_2.shape[2], aug_data.shape[3], 2))

# Assign the data from array1 to the new array
padded_aug[:, :, :aug_data.shape[2], :, 0] = aug_data[:, :, :, :]
del aug_data
# Concatenate both arrays along the first axis
x_train_concat = np.concatenate((data_orig_train_2, padded_aug), axis=0) # N, C, T, V, M
y_train_concat = np.concatenate((label_orig_train, aug_label), axis=0)
N, C, T, V, M = x_train_concat.shape
x_train_trans = x_train_concat.transpose(0, 2, 4, 3, 1) # N, T, M, V, C
del x_train_concat, padded_aug, label_orig_train, aug_label, data_orig_train_2


x_train = x_train_trans.reshape(N, T, M*V*C)
del x_train_trans
train_y = one_hot_vector(y_train_concat)

npz_data = np.load(arg.original)
test_x, test_y = npz_data['x_test'], npz_data['y_test']
del npz_data
np.savez('NTU120_CSub.npz', x_train=x_train, y_train= train_y, x_test=test_x, y_test=test_y)
# print(data_matches)

print('Process Complete...')
