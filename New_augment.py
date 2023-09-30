import numpy as np
import torch
import torch.nn.functional as F
import argparse
import pickle
from demo_object import *
from tqdm import tqdm
from Skeleton_Normalization.skeleton_normalization import skeleton_normalization

parser = argparse.ArgumentParser(description='JFT augmentation and make phase, amplitude components')
parser.add_argument('-d', '--datapath', default='/home/ntkkh958/My_work/Dataset_preparation/Raw_datasets/NTU15/NTU120_15sub_CSub.npz',
                    help='location of dataset npz file')
parser.add_argument('-n', '--outdatapath_normal', default='./All_Normalised_data_ntu15_both_train_test.npz',
                    help='location of dataset npz file')
parser.add_argument('-f', '--outdatapath_fourier', default='./Fourier_data_all_normalised_ntu15_both_train_test.npz',
                    help='location of dataset npz file')
parser.add_argument('-b', '--both_norm', default='y')
parser.add_argument('-p', '--p_intervel', default=[0.95])
parser.add_argument('-w', '--window_size', default=74)  #82
# parser.add_argument('-v', '--visual_frame', default=16)
parser.add_argument('-j', '--joints', default=25)


def load_data(data):
    npz_data = np.load(data)
    data, label, test_data, label_test = npz_data['x_train'], npz_data['y_train'], npz_data['x_test'], npz_data['y_test']
    del npz_data
    N, T, _ = data.shape
    N1, T1, _ = test_data.shape
    data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N C T V M
    data_test = test_data.reshape((N1, T1, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N C T V M
    return data, label, data_test, label_test


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


def crop_dataset(data, p_intervel, window_size):
    all_seq = []
    for index in tqdm(range(data.shape[0])):  # range(data.shape[0])
        data_numpy = data[index]  # Get the data_numpy for the current index.
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, p_intervel, window_size)
        all_seq.append(data_numpy)
    return np.array(all_seq)  # N, C, T, V, M


def get_graph_spectral(bone_pair, num_joint):
    A = np.zeros((num_joint, num_joint))
    for i, j in bone_pair:
        A[i, j] = 1
        A[j, i] = 1

    D = np.diag(A.sum(axis=1))
    L = D - A
    Lam, U = np.linalg.eigh(L)
    idx = Lam.argsort()
    P = Lam[idx]
    Q = U.T[idx].T
    return P, Q


def get_spatial_spectral(skes_data, sub_num, Q):
    # input C T V
    spectral_dict_1 = dict()
    for c in range(1, 4):
        spectral_dict_1['ax_{:d}'.format(c)] = list()

    if sub_num == 1:
        data = np.copy(skes_data)
        for t in range(data.shape[1]):
            for c in range(1, 4):
                spectral_dict_1['ax_{:d}'.format(c)].append(np.dot(Q.T, data[c - 1, t]))

        for c in range(1, 4):
            spectral_dict_1['ax_{:d}'.format(c)] = np.stack(spectral_dict_1['ax_{:d}'.format(c)])

        return spectral_dict_1

    else:
        spectral_dict_2 = dict()
        for c in range(1, 4):
            spectral_dict_2['ax_{:d}'.format(c)] = list()

        spectral_dicts = [spectral_dict_1, spectral_dict_2]
        for m in range(2):
            data = np.copy(skes_data[m])
            for t in range(data.shape[1]):
                for c in range(1, 4):
                    spectral_dicts[m]['ax_{:d}'.format(c)].append(np.dot(Q.T, data[c - 1, t]))

        for m in range(2):
            for c in range(1, 4):
                spectral_dicts[m]['ax_{:d}'.format(c)] = np.stack(spectral_dicts[m]['ax_{:d}'.format(c)])

        return spectral_dicts


def get_temporal_spectral(skes_data, sub_num):
    # input C T V
    spectral_dict_1 = dict()

    data = np.copy(skes_data)
    N = data.shape[-2]

    if sub_num == 1:
        data = data.transpose(0, 2, 1)  # C V T
        for c in range(1, 4):
            amp = np.fft.fft(data[c - 1, :, :], axis=-1)
            spectral_dict_1['ax_{:d}'.format(c)] = amp.copy()
        return spectral_dict_1
    else:
        data = data.transpose(0, 1, 3, 2)  # M C V T
        spectral_dict_2 = dict()
        spectral_dicts = [spectral_dict_1, spectral_dict_2]
        for m in range(2):
            for c in range(1, 4):
                amp = np.fft.fft(data[m, c - 1, :, :], axis=-1)
                spectral_dicts[m]['ax_{:d}'.format(c)] = amp.copy()
        return spectral_dicts


def inverse_fourier_transform(freq_spectrum, U, min_val, max_val):
    data_augemented = []
    freq_spec = np.array(freq_spectrum)
    min_val=np.array(min_val).transpose(0, 1, 3, 2) #in M C 1 V out M C V 1
    max_val=np.array(max_val).transpose(0, 1, 3, 2) ##in M C 1 V out M C V 1
    for i in range(freq_spec.shape[0]):
        inter_aug = []
        for c in range(3):
            check_frq_spectrum = freq_spectrum[i][c]
            spatial_spectral= np.real(np.fft.ifft(freq_spectrum[i][c], axis=-1))
            spatial_spectral = min_val[i, c, :, :] + \
                                   (spatial_spectral + 1) * 0.5 * (max_val[i, c, :, :] - min_val[i, c, :, :])
            # spatial_spectral_real = np.abs(np.fft.ifft(freq_spectrum[i][c], axis=-1))
            # spatial_spectral_img = np.fft.ifft(freq_spectrum[i][c], axis=-1).imag
            coordinates = np.dot(spatial_spectral.T, U.T) #
            inter_aug.append(coordinates)
        data_augemented.append(inter_aug)
    result_aug = np.array(data_augemented)
    return result_aug


def fft_augment(data, bone_pairs, num_joint):
    P, Q = get_graph_spectral(bone_pairs, num_joint)
    amplitude = []
    phase = []
    max_joint = []
    min_joint = []

    N, C, T, V, M = data.shape
    # T = arg.visual_frame

    for index in tqdm(range(data.shape[0])):  # data.shape[0]


        data_numpy = data[index]
        # C, T, V, M = data_numpy.shape
        # valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(
        #     -1) != 0)  # #Compute the valid_frame_num based on the non-zero values in data_numpy.
        # data_numpy = valid_crop_resize(data_numpy, valid_frame_num, [0.95],
        #                                arg.visual_frame)  # #Call the valid_crop_resize function to crop and resize the data.

        C, T, V, M = data_numpy.shape
        data_numpy = data_numpy.transpose(3, 0, 1, 2).reshape(M, C, T, V)  # Reshape and transpose the data_numpy array

        if (data_numpy[1] == np.zeros((C, T, V), np.float32)).all():
            num_sub = 1
            phase_component = np.zeros((M, C, V, T), dtype=np.float64)
            amplitude_component = np.zeros((M, C, V, T), dtype=np.float64)
            min_joint_values = np.zeros((M, C, 1, V), dtype=np.float64)
            max_joint_values = np.zeros((M, C, 1, V), dtype=np.float64)
            spatial_spectral = get_spatial_spectral(skes_data=data_numpy[0], sub_num=1, Q=Q)
            new_sp = np.stack(spatial_spectral.values())
            min_values = np.min(new_sp, axis=1, keepdims=True)
            max_values = np.max(new_sp, axis=1, keepdims=True)
            spatial_spectral = -1 + 2 * (new_sp - min_values) / (max_values - min_values)
            spatial_temporal_spectral = get_temporal_spectral(spatial_spectral, sub_num=1)

            for c in range(1, 4):   #Compute the absolute values of the spectral features.
                # spatial_temporal_phase['ax_{:d}'.format(c)] = np.angle(spatial_temporal_spectral['ax_{:d}'.format(c)])
                # spatial_temporal_amp['ax_{:d}'.format(c)] = np.abs(spatial_temporal_spectral['ax_{:d}'.format(c)])
                phase_component[0, c - 1, :, :] = np.angle(spatial_temporal_spectral['ax_{:d}'.format(c)])
                amplitude_component[0, c - 1, :, :] = np.abs(spatial_temporal_spectral['ax_{:d}'.format(c)])
            # phase_component[0, :, :, :] = np.stack([spatial_temporal_phase[key] for key in spatial_temporal_phase], axis=0)
            # amplitude_component[0, :, :, :] = np.stack([spatial_temporal_amp[key] for key in spatial_temporal_amp], axis=0)
            min_joint_values[:, :, :, :] = min_values
            max_joint_values[:, :, :, :] = max_values
        else:
            phase_component = np.zeros((M, C, V, T), dtype=np.float64)
            amplitude_component = np.zeros((M, C, V, T), dtype=np.float64)
            min_joint_values = np.zeros((M, C, 1, V), dtype=np.float64)
            max_joint_values = np.zeros((M, C, 1, V), dtype=np.float64)
            num_sub = 2
            # if index == 111:
            #     data_numpy[1, :, :, :] = data_numpy[0]
            spatial_spectral_2 = get_spatial_spectral(skes_data=data_numpy, sub_num=2, Q=Q)
            input_spatial_spectral = list()

            for m in range(2):
                for c in range(1, 4):
                    input_spatial_spectral.append(spatial_spectral_2[m]['ax_{:d}'.format(c)])
            input_spatial_spectral = np.stack(input_spatial_spectral).reshape(2, C, T, V)   #2 C T V

            min_values = np.min(input_spatial_spectral, axis=2, keepdims=True)  # M C 1 V
            max_values = np.max(input_spatial_spectral, axis=2, keepdims=True)  # M C 1 V
            input_spatial_spectral = -1 + 2 * (input_spatial_spectral - min_values) / (max_values - min_values)
            spatial_temporal_spectral_2 = get_temporal_spectral(skes_data=input_spatial_spectral, sub_num=2)
            min_joint_values[:, :, :, :] = min_values
            max_joint_values[:, :, :, :] = max_values
            for m in range(2):
                for c in range(1, 4):
                    # spatial_temporal_phase['ax_{:d}'.format(c)] = np.angle(spatial_temporal_spectral_2[m]['ax_{:d}'.format(c)])
                    # spatial_temporal_amp['ax_{:d}'.format(c)] = np.abs(spatial_temporal_spectral_2[m]['ax_{:d}'.format(c)])
                    # test = np.angle(spatial_temporal_spectral_2[m]['ax_{:d}'.format(c)])
                    phase_component[m, c-1, :, :] = np.angle(spatial_temporal_spectral_2[m]['ax_{:d}'.format(c)])
                    amplitude_component[m, c-1, :, :] = np.abs(spatial_temporal_spectral_2[m]['ax_{:d}'.format(c)])
            s12 = amplitude_component * (np.e ** (1j * phase_component))
            ift_ = inverse_fourier_transform(s12, Q, min_joint_values, max_joint_values)  # M C T V
            test = ift_.transpose(1, 2, 3, 0 )# C T V M

        amplitude.append(np.copy(amplitude_component)) # N, M, C, V, T
        phase.append(np.copy(phase_component))   # N M C V T
        max_joint.append(np.copy(max_joint_values))  # N, M, C, 1, V
        min_joint.append(np.copy(min_joint_values))  # N, M, C, 1, V
    # print('Done')
    amp, phase, max_, min_ = np.array(amplitude), np.array(phase), np.array(max_joint), np.array(min_joint)
    return amp, phase, max_, min_


if __name__ == '__main__':
    arg = parser.parse_args()
    data_train, label_train, data_test, label_test = load_data(arg.datapath)  # N C T V M
    print("start Cropping train dataset to average valid frame length(training)")
    crop_data = crop_dataset(data_train, arg.p_intervel, arg.window_size)  # N, C, T, V, M
    del data_train
    print("Start normalisation(training)")
    data_trans = crop_data.transpose(0, 4, 2, 3, 1)  # N M T V C
    data_normalised = skeleton_normalization(data_trans)  # N M T V C
    data_normalised = np.array(data_normalised)
    N, M, T, V, C = data_normalised.shape
    data_norm_train = data_normalised.transpose(0, 2, 1, 3, 4).reshape(N, T, C * V * M)

    if arg.both_norm == 'y':
        print("start Cropping test dataset to average valid frame length(test)")
        crop_data_test = crop_dataset(data_test, arg.p_intervel, arg.window_size)  # N, C, T, V, M
        del data_test
        print("Start normalisation(testing)")
        data_trans_test = crop_data_test.transpose(0, 4, 2, 3, 1)  # N M T V C
        data_normalised_test = skeleton_normalization(data_trans_test)  # N M T V C
        data_normalised_test = np.array(data_normalised_test)
        N1, M1, T1, V1, C1 = data_normalised_test.shape
        data_norm_test = data_normalised_test.transpose(0, 2, 1, 3, 4).reshape(N1, T1, C1 * V1 * M1)
    else:
        data_test_trans = data_test.transpose(0, 4, 2, 3, 1)
        del data_test
        N1, M1, T1, V1, C1 = data_test_trans.shape
        data_norm_test = data_test_trans.transpose(0, 2, 1, 3, 4).reshape(N1, T1, C1 * V1 * M1)
    print("save normalised file")
    np.savez(arg.outdatapath_normal, x_train=data_norm_train, y_train=label_train, x_test=data_norm_test, y_test=label_test)
    del data_norm_train, label_train, data_norm_test, label_test
    # '''
    np_seq = data_normalised.transpose(0, 4, 2, 3, 1)  # N, C, T, V, M
    print('start augmentation')
    amplitude, phase, max_joint, min_joint = fft_augment(np_seq, ntu_skeleton_bone_pairs, arg.joints)
    print('Create file')
    np.savez(arg.outdatapath_fourier, JFT_amplitude=amplitude, JFT_phase=phase, JFT_min_joint=min_joint, JFT_max_joint=max_joint)
    print('File creation Done..')
    # '''
