import numpy as np
import argparse
from Skeleton_Normalization.skeleton_normalization import skeleton_normalization
import pickle

parser = argparse.ArgumentParser(description='Visualization NTU RGB+D')
parser.add_argument('--datapath', default='./Cropped_data.pkl')
parser.add_argument('--outfile_name', default='Skeleton_Normalise.pkl')
parser.add_argument('--extention', default='.pkl', choices=['.pkl', '.npz'])
parser.add_argument('--outpath', default='./')


arg = parser.parse_args()
if arg.extention == '.npz':
    npz_data = np.load(arg.datapath)
    data, label = npz_data['x_test'], np.where(npz_data['y_test'] > 0)[1]
    del npz_data

    N, T, _ = data.shape
    data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N C T V M
    data = data.transpose(0, 4, 2, 3, 1)    # N M T V C
    print(data.shape)
else:
    with open(arg.datapath, 'rb') as file:
        data = np.array(pickle.load(file))  #N, C, T, V, M
        data = data.transpose(0, 4, 2, 3, 1)
data = skeleton_normalization(data)
np_seq = np.array(data).transpose(0, 4, 2, 3, 1)  # N, C, T, V, M


with open(arg.outfile_name, 'wb') as f:
    pickle.dump(np_seq, f)