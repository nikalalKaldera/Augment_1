import numpy as np
import torch
import torch.nn.functional as F
import argparse
import pickle
from demo_object import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Frequency analysise')

parser.add_argument('-p', '--datapath', default='/home/ntkkh958/My_work/Codes/CTR_NEW_26/CTR-GCN/data/ntu/NTU60_CS.npz',
                    help='location of dataset npz file')
parser.add_argument('--extention', default='.npz', choices=['.pkl', '.npz'])
parser.add_argument('-d', '--dataset',
                    choices=['NTU60 CS', 'NTU60 CV', 'NTU120_CSet', 'NTU120_CSub', 'HDM05'],
                    default='NTU60 CS')




def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64),
                                    valid_size)  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',
                         align_corners=False).squeeze()  # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


if __name__ == '__main__':
    arg = parser.parse_args()  # Parse the command-line arguments.
    p_interval = [0.95]  # Set values for p_interval, window_size, data, label, N, and data.
    window_size = 64

    if arg.extention == '.npz':
        npz_data = np.load(arg.datapath)
        data, label = npz_data['x_test'], np.where(npz_data['y_test'] > 0)[1]
        del npz_data
        N, T, _ = data.shape
        data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N C T V M
        data = data.transpose(0, 1, 2, 3, 4)    # N M T V C
    else:
        with open(arg.datapath, 'rb') as file:
            data = np.array(pickle.load(file))
            data = data.transpose(0, 4, 2, 3, 1)

    if arg.dataset[:3] == 'NTU':
        bone_pairs = ntu_skeleton_bone_pairs
        orderd_bone_pairs = ntu_pairs
        num_joint = 25
    elif arg.dataset[:3] == 'HDM':
        pass
    else:
        raise Exception('Dataset is not accurate.')
    all_seq = []

    for index in tqdm(range(data.shape[0])): #data.shape[0]
        data_numpy = data[index]    #Get the data_numpy for the current index.
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)    #Compute the valid_frame_num based on the non-zero values in data_numpy.
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, p_interval, window_size)    #Call the valid_crop_resize function to crop and resize the data.
        all_seq.append(data_numpy)

    np_seq = np.array(all_seq)  # N, C, T, V, M
    # label_reshaped = label[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    # result_array = np.append(np_seq, label_reshaped, axis=1)

    with open('Cropped_data.pkl', 'wb') as f:
        pickle.dump(np_seq, f)
    with open('label.pkl', 'wb') as f:
        pickle.dump(label, f)