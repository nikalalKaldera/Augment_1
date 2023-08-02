import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import imageio
import glob
import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Visualization NTU RGB+D')
parser.add_argument('--extention', default='.pkl', choices=['.pkl', '.npz'])
# parser.add_argument('--datapath', default='/home/ntkkh958/My_work/Codes/InfoGCN/infogcn/data/ntu/NTU60_CS.npz')
# parser.add_argument('--datapath', default='./Skeleton_Normalise.pkl')
parser.add_argument('--datapath', default='./Augmented_sequence.pkl')
# parser.add_argument('--datapath', default='./Skeleton_Normalise.pkl')

parser.add_argument('--vis', default='3D', choices=['2D', '3D'])
parser.add_argument('--skel_no', default=0)
parser.add_argument('--save_directory', default='/home/ntkkh958/My_work/Codes/GFT/output_file/')

trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]
joint_data = []
joint_data_y = []
joint_data_z = []
num_joint = 25
max_frame = 300
max_body_true = 2


# Show 3D Skeleton with Axes3D for NTU RGB+D
class Draw3DSkeleton(object):

    def __init__(self, file, save_path=None, init_horizon=-45,
                 init_vertical=20, x_rotation=None,
                 y_rotation=None, pause_step=0.2):

        self.file = file
        self.save_path = save_path

        #  if not os.path.exists(self.save_path):
        #      os.mkdir(self.save_path)

        # self.xyz = self.read_xyz(self.file)
        self.init_horizon = init_horizon
        self.init_vertical = init_vertical

        self.x_rotation = x_rotation
        self.y_rotation = y_rotation

        self._pause_step = pause_step

    def _normal_skeleton(self, data):
        #  use as center joint
        center_joint = data[:, 0, :]
        # print(center_joint)

        center_jointx = np.mean(center_joint[:, 0])
        center_jointy = np.mean(center_joint[:, 1])
        center_jointz = np.mean(center_joint[:, 2])

        center = np.array([center_jointx, center_jointy, center_jointz])
        data = data - center

        return data

    def _rotation(self, data, alpha=0, beta=0):
        # rotate the skeleton around x-y axis
        r_alpha = alpha * np.pi / 180
        r_beta = beta * np.pi / 180

        rx = np.array([[1, 0, 0],
                       [0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
                       [0, np.sin(r_alpha), np.cos(r_alpha)]]
                      )

        ry = np.array([
            [np.cos(r_beta), 0, np.sin(r_beta)],
            [0, 1, 0],
            [-1 * np.sin(r_beta), 0, np.cos(r_beta)],
        ])

        r = ry.dot(rx)
        data = data.dot(r)

        return data

    def visual_skeleton(self):
        fig = plt.figure(dpi=180)
        ax = Axes3D(fig)

        ax.view_init(self.init_vertical, self.init_horizon)
        plt.ion()

        # print(self.xyz.shape)

        data = np.transpose(self.file, (1, 2, 0))   #input shape: C T V output: T V C
        print('data = ', data.shape)
        # data rotation
        if (self.x_rotation is not None) or (self.y_rotation is not None):

            if self.x_rotation > 180 or self.y_rotation > 180:
                raise Exception("rotation angle should be less than 180.")

            else:
                data = self._rotation(data, self.x_rotation, self.y_rotation)

        # data normalization
        data = self._normal_skeleton(data)

        # show every frame 3d skeleton
        for frame_idx in tqdm(range(data.shape[0])): #data.shape[0])
        # for frame_idx in range(data.shape[0]):

            plt.cla()
            plt.title("Frame: {}".format(frame_idx))

            ax.set_xlim3d([-1, 1])
            ax.set_ylim3d([-1, 1])
            ax.set_zlim3d([-0.8, 0.8])

            x = data[frame_idx, :, 0]
            y = data[frame_idx, :, 1]
            z = data[frame_idx, :, 2]

            # print(type(x))

            for part in body:
                x_plot = x[part]
                y_plot = y[part]
                z_plot = z[part]
                ax.plot(x_plot, z_plot, y_plot, color='b', marker='o', markerfacecolor='r')

            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')

            if self.save_path is not None:
                save_pth = os.path.join(self.save_path, '{}.png'.format(frame_idx))
                plt.savefig(save_pth)
            # print("The {} frame 3d skeleton......".format(frame_idx))

            ax.set_facecolor('none')
            # ax.view_init(elev = 109, azim=-92)
            # plt.pause(self._pause_step)

        plt.ioff()
        ax.axis('off')
        # plt.show()

        os.chdir(self.save_path)
        # print('save pathhh ==============' , self.save_path)
        # name = os.path.splitext(self.file)[0]
        # name = os.path.splitext(os.path.basename(self.file))[0]
        name = 'test'
        gif_path = os.path.join(self.save_path, str(name) + ".gif")

        with imageio.get_writer(gif_path, mode='I') as writer:
            for i in range(data.shape[0]):   # data.shape[0]
                save_pth = os.path.join(self.save_path, '{}.png'.format(i))
                writer.append_data(imageio.imread(save_pth))
        for file_name in glob.glob("*.png"):
            os.remove(file_name)
        print('Completed-------')


if __name__ == '__main__':
    arg = parser.parse_args()
    if arg.extention == '.npz':
        npz_data = np.load(arg.datapath)
        data, label = npz_data['x_test'], np.where(npz_data['y_test'] > 0)[1]
        del npz_data
        N, T, _ = data.shape
        data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N C T V M
        data = data.transpose(0, 1, 2, 3, 4)  # N M T V C
    else:
        with open(arg.datapath, 'rb') as file:
            data = np.array(pickle.load(file))  # N, C, T,V
            # data = data.transpose(0, 4, 2, 3, 1) #C, T, V
    # print(data.shape)
    # data = data[arg.skel_no][0].transpose(2, 1, 0)
    # print(data.shape)
    data = data[arg.skel_no]
    # new_dat = data[:, :, :, 0]
    # sk = Draw3DSkeleton(data[:, :, :, 0], arg.save_directory)
    sk = Draw3DSkeleton(data, arg.save_directory)
    sk.visual_skeleton()
