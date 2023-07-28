import numpy as np
import torch
import torch.nn.functional as F
import argparse
import pickle
from demo_object import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Frequency analysise')

parser.add_argument('-p', '--datapath', default='/home/ntkkh958/My_work/Codes/InfoGCN/infogcn/data/ntu/NTU60_CS.npz',
                    help='location of dataset npz file')
parser.add_argument('--extention', default='.npz', choices=['.pkl', '.npz'])

parser.add_argument('-d', '--dataset',
                    choices=['NTU60 CS', 'NTU60 CV', 'NTU120_CSet', 'NTU120_CSub', 'HDM05'],
                    default='NTU60 CS')




def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    '''Define the valid_crop_resize function to perform data cropping and resizing.
    Takes the data_numpy array, valid_frame_num, p_interval, and window as inputs.
    Extracts the shape of the data_numpy array and sets the initial values for cropping.
    If p_interval has only one value, perform a center crop based on the valid_size.
    If p_interval has two values, randomly select a p value within the range and crop the data accordingly.
    Resize the data using F.interpolate from torch library.
    Return the resized data.'''

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
                check = T
                check_QT = Q.T
                check_data = data[c - 1, t]
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


def inverse_fourier_transform(freq_spectrum, U):
    data_augemented = []
    freq_spec = np.array(freq_spectrum)
    for i in range(freq_spec.shape[0]):
        inter_aug = []
        for c in range(3):
            spatial_spectral = np.fft.ifft(freq_spectrum[i][c], axis=-1).real
            coordinates = np.dot(spatial_spectral.T, U.T)
            inter_aug.append(coordinates)
        data_augemented.append(inter_aug)
    result_aug = np.array(data_augemented)
    return result_aug


def test_fft_inv(magnitude, phase, U, data):
    complex_coeffs = magnitude * np.exp(1j * phase)
    # temporal IFFT
    spatial_spectral = np.fft.ifft(complex_coeffs, axis=-1).real
    # spatial_spectral = spatial_spectral.transpose(0, 1, 3, 2)
    data_numpy = np.dot(spatial_spectral.T, U.T)
    # spectral_dict_1['ax_{:d}'.format(c)].append(np.dot(Q.T, data[c - 1, t]))
    # data_numpy = data_numpy.transpose(1, 2, 3, 0)
    print(data_numpy.shape)
    return data_numpy

def cross_mix_amplitude_phase(skeleton_magnitude, skeleton_phase, x):
    N, C, V, T = skeleton_magnitude.shape

    # Initialize a list to store the augmented sequences
    augmented_sequences = []

    for _ in range(x):
        # Randomly select two different samples (N) to cross-mix
        sample_idx1, sample_idx2 = 1, 1

        # Extract magnitude and phase for the selected samples
        magnitude_sample1 = skeleton_magnitude[sample_idx1]
        phase_sample2 = skeleton_phase[sample_idx2]

        # Cross-mix amplitude and phase
        s12 = magnitude_sample1 * np.exp(1j * phase_sample2)

        # Create the augmented sequence
        augmented_sequences.append(s12)

    return augmented_sequences


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

    P, Q = get_graph_spectral(bone_pairs, num_joint)
    spatial_dict = dict()
    temporal_dict = dict()
    spatial_temporal_dict = dict()
    temporal_spatial_dict = dict()
    spatial_temporal_phase_dict = dict()
    spatial_temporal_phase = dict()
    spatial_temporal_amp = dict()
    spatial_temporal_amp_dict = dict()
    resultant_array_mag = np.zeros((50, data.shape[1], data.shape[3], window_size))
    resultant_array_phase = np.zeros((50, data.shape[1], data.shape[3], window_size))
    check_data_np = []

    num_sub = 0     #Initialize a counter num_sub.

    for c in range(1, 4):
        spatial_dict['ax_{:d}'.format(c)] = 0.0
        temporal_dict['ax_{:d}'.format(c)] = 0.0
        spatial_temporal_dict['ax_{:d}'.format(c)] = 0.0
        spatial_temporal_phase_dict['ax_{:d}'.format(c)] = 0.0

    for index in tqdm(range(50)):
        data_numpy = data[index]    #Get the data_numpy for the current index.
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)    #Compute the valid_frame_num based on the non-zero values in data_numpy.
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, p_interval, window_size)    #Call the valid_crop_resize function to crop and resize the data.

        C, T, V, M = data_numpy.shape
        data_numpy = data_numpy.transpose(3, 0, 1, 2).reshape(M, C, T, V)   #Reshape and transpose the data_numpy array.
        check_data_np.append(data_numpy[0])
        if (data_numpy[1] == np.zeros((C, T, V), np.float32)).all():    #Check if the second element in data_numpy is all zeros to determine the number of subjects.
            num_sub += 1    #If there is only one subject, compute the spatial, temporal, and spatial-temporal spectral features.
            spatial_spectral = get_spatial_spectral(skes_data=data_numpy[0], sub_num=1, Q=Q)
            spatial_temporal_spectral = get_temporal_spectral(skes_data=np.stack(spatial_spectral.values()), sub_num=1)

            for c in range(1, 4):   #Compute the absolute values of the spectral features.
                spatial_temporal_phase['ax_{:d}'.format(c)] = np.angle(spatial_temporal_spectral['ax_{:d}'.format(c)])
                # spatial_temporal_amp['ax_{:d}'.format(c)] = np.abs(spatial_temporal_spectral['ax_{:d}'.format(c)])
                spatial_temporal_spectral['ax_{:d}'.format(c)] = np.abs(spatial_temporal_spectral['ax_{:d}'.format(c)])
                # check_invfft = test_fft_inv(spatial_temporal_amp['ax_{:d}'.format(c)],
                #                                          spatial_temporal_phase['ax_{:d}'.format(c)], Q, data_numpy[0])

        else:
            num_sub += 2
            spatial_spectral_2 = get_spatial_spectral(skes_data=data_numpy, sub_num=2, Q=Q)
            input_spatial_spectral = list()
            for m in range(2):
                for c in range(1, 4):
                    input_spatial_spectral.append(spatial_spectral_2[m]['ax_{:d}'.format(c)])
            input_spatial_spectral = np.stack(input_spatial_spectral).reshape(2, C, T, V)
            spatial_temporal_spectral_2 = get_temporal_spectral(skes_data=input_spatial_spectral, sub_num=2)

            spatial_spectral = dict()
            spatial_temporal_spectral = dict()

            for c in range(1, 4):
                spatial_spectral['ax_{:d}'.format(c)] = np.abs(spatial_spectral_2[0]['ax_{:d}'.format(c)]) + np.abs(
                    spatial_spectral_2[1]['ax_{:d}'.format(c)])
                spatial_temporal_phase['ax_{:d}'.format(c)] = np.angle(
                    spatial_temporal_spectral_2[0]['ax_{:d}'.format(c)]) + np.angle(
                    spatial_temporal_spectral_2[1]['ax_{:d}'.format(c)])
                spatial_temporal_spectral['ax_{:d}'.format(c)] = np.abs(
                    spatial_temporal_spectral_2[0]['ax_{:d}'.format(c)]) + np.abs(
                    spatial_temporal_spectral_2[1]['ax_{:d}'.format(c)])


        for k in spatial_dict.keys():   #Update the spatial, temporal, and spatial-temporal dictionaries with the computed spectral features.
            spatial_dict[k] += spatial_spectral[k]
            spatial_temporal_dict[k] += spatial_temporal_spectral[k]
            spatial_temporal_phase_dict[k] += spatial_temporal_phase[k]

        for k in spatial_dict.keys():  # Normalize the computed spectral features by dividing them by num_sub.
            spatial_dict[k] /= num_sub
            spatial_temporal_dict[k] /= num_sub
            spatial_temporal_phase_dict[k] /= num_sub

        st_mag1 = spatial_temporal_dict['ax_1']
        st_mag2 = spatial_temporal_dict['ax_2']
        st_mag3 = spatial_temporal_dict['ax_3']
        st_mag_array = np.array([st_mag1, st_mag2, st_mag3])
        resultant_array_mag[index] = st_mag_array       # N, C, V, T

        st_phase1 = spatial_temporal_phase_dict['ax_1']
        st_phase2 = spatial_temporal_phase_dict['ax_2']
        st_phase3 = spatial_temporal_phase_dict['ax_3']
        st_phase_array = np.array([st_phase1, st_phase2, st_phase3])
        resultant_array_phase[index] = st_phase_array  # N, C, V, T

    mix = cross_mix_amplitude_phase(resultant_array_mag, resultant_array_phase, 1)
    raw_data_check = np.array(check_data_np)
    ift_ = inverse_fourier_transform(mix, Q)

    with open('Augmented_sequence.pkl', 'wb') as f:
        pickle.dump(ift_, f)

    with open('Raw_data.pkl', 'wb') as f:
        pickle.dump(raw_data_check, f)