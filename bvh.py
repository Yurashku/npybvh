# -*- coding: utf-8 -*-
# +
import numpy as np
import re
import pickle
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation

def centering(poses, coefs=(5, 1, 5)):
    limits = []
    for ax_n in range(3):
        l, r = np.min(poses[..., ax_n]), np.max(poses[..., ax_n])
        center = (l + r)/2
        rad = max((r - l)/2, 1)
        limits.append([center - rad * coefs[ax_n], center + rad * coefs[ax_n]])
    return np.array(limits)


def pair2matrix(v_start, v_finish):
    '''
    Вычисляет матрицу кратчайшего поворота, сопоставляющий два вектора
    '''
    v_start = v_start / np.linalg.norm(v_start)
    v_finish = v_finish / np.linalg.norm(v_finish)
    theta = np.arccos(np.clip(np.dot(v_start, v_finish), -1.0, 1.0))
    
    quat_copmponents = np.zeros(4)
    quat_copmponents[3] = np.cos(theta / 2)
    n = np.cross(v_start, v_finish)
    n = n / np.linalg.norm(n)
    quat_copmponents[:3] = n * np.sin(theta / 2)
    
    r = Rotation.from_quat(quat_copmponents)
    return r.as_matrix()
        

def get_matrix_rotation(joint, frame_number, p, joint_name_index):
    '''
    Вычисляет матрицу поворота, соответствующую заданной вершине joint
    '''
    parent = joint
    children = parent.children
    
    starts = np.array([ch.offset for ch in children])
    finishes = np.array([p[frame_number, joint_name_index[ch.name]] -
                         p[frame_number, joint_name_index[parent.name]]
                         for ch in children])
    
    
    if len(children) > 1:
        rot = Rotation.align_vectors(starts, finishes)[0]
        total_child_matrix = rot.inv().as_matrix()
        
    elif len(children) == 1:
#         print('rotation with 1 vector', joint.__repr__())
        if np.linalg.norm(finishes[0]) < 0.0001:
            total_child_matrix = np.eye(3, dtype=float)
        else:
            total_child_matrix = pair2matrix(starts[0], finishes[0])
    else:
#         print('0-dimension children pool in ' + joint.__repr__())
        total_child_matrix = np.eye(3, dtype=float)
    
    
    return total_child_matrix



class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return 'Joint_' + self.name

    def position_animated(self):
        return any([x.endswith('position') for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith('rotation') for x in self.channels])

def _recursive_write_hierarchy(file, joint, degree):
    if degree == 0:
        name = 'ROOT ' + joint.name
    elif joint.name[-4:] == '_end':
        name = 'End Site'
    else:
        name = 'JOINT ' + joint.name
    file.write('\t' * degree + name + '\n')
    file.write('\t' * degree + '{\n')
    tmp = ' '.join([f'{el:f}' for el in joint.offset])
    file.write('\t' * (degree + 1) + 'OFFSET ' + tmp + '\n')
    tmp = ' '.join(joint.channels)
    if degree==0:
        file.write('\t' * (degree + 1) + f'CHANNELS {len(joint.channels)} ' + tmp + ' \n')
    elif joint.name[-4:] == '_end':
        ...
    else:
        file.write('\t' * (degree + 1) + f'CHANNELS {len(joint.channels)} ' + tmp + '\n')
    for children in joint.children:
        _recursive_write_hierarchy(file, children, degree + 1)
    file.write('\t' * degree + '}\n')

class Bvh:
    def __init__(self):
        self.joints = {}
        self.root = None
        self.keyframes = None
        self.frames = 0
        self.fps = 0

    def _parse_hierarchy(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        joint_stack = []

        for line in lines:
            words = re.split('\\s+', line)
            instruction = words[0]

            if instruction == "JOINT" or instruction == "ROOT":
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent)
                self.joints[joint.name] = joint
                if parent:
                    parent.add_child(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
            elif instruction == "CHANNELS":
                for i in range(2, len(words)):
                    joint_stack[-1].channels.append(words[i])
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1])
                joint_stack[-1].add_child(joint)
                joint_stack.append(joint)
                self.joints[joint.name] = joint
            elif instruction == '}':
                joint_stack.pop()

    def _add_pose_recursive(self, joint, offset, poses):
        pose = joint.offset + offset
        poses.append(pose)

        for c in joint.children:
            self._add_pose_recursive(c, pose, poses)

    def plot_hierarchy(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        poses = []
        self._add_pose_recursive(self.root, np.zeros(3), poses)
        pos = np.array(poses)
        
        limits = centering(pos)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[2])
        ax.set_zlim(*limits[1])
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1])
        plt.show()


    def get_initial_poses(self):
        poses = []
        self._add_pose_recursive(self.root, np.zeros(3), poses)
        pos = np.array(poses)
        
        return pos

    def parse_motion(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        frame = 0
        for line in lines:
            if line == '':
                continue
            words = re.split('\\s+', line)

            if line.startswith("Frame Time:"):
                self.fps = round(1 / float(words[2]))
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue

            if self.keyframes is None:
                self.keyframes = np.empty((self.frames, len(words)), dtype=np.float32)

            for angle_index in range(len(words)):
                self.keyframes[frame, angle_index] = float(words[angle_index])

            frame += 1

    def parse_string(self, text):
        hierarchy, motion = text.split("MOTION")
        self._parse_hierarchy(hierarchy)
        self.parse_motion(motion)

    def _extract_rotation(self, frame_pose, index_offset, joint):
        local_rotation = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            if channel == "Xrotation":
                local_rotation[0] = frame_pose[index_offset]
            elif channel == "Yrotation":
                local_rotation[1] = frame_pose[index_offset]
            elif channel == "Zrotation":
                local_rotation[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        local_rotation = np.deg2rad(local_rotation)
        M_rotation = np.eye(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue

            if channel == "Xrotation":
                euler_rot = np.array([local_rotation[0], 0., 0.])
            elif channel == "Yrotation":
                euler_rot = np.array([0., local_rotation[1], 0.])
            elif channel == "Zrotation":
                euler_rot = np.array([0., 0., local_rotation[2]])
            else:
                raise Exception(f"Unknown channel {channel}")

            M_channel = euler2mat(*euler_rot)
            M_rotation = M_rotation.dot(M_channel)

        return M_rotation, index_offset

    def _extract_position(self, joint, frame_pose, index_offset):
        offset_position = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("rotation"):
                continue
            if channel == "Xposition":
                offset_position[0] = frame_pose[index_offset]
            elif channel == "Yposition":
                offset_position[1] = frame_pose[index_offset]
            elif channel == "Zposition":
                offset_position[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1

        return offset_position, index_offset

    def _recursive_apply_frame(self, joint, frame_pose, index_offset, p, r, M_parent, p_parent):
        if joint.position_animated():
            offset_position, index_offset = self._extract_position(joint, frame_pose, index_offset)
        else:
            offset_position = np.zeros(3)

        if len(joint.channels) == 0:
            joint_index = list(self.joints.values()).index(joint)
            p[joint_index] = p_parent + M_parent.dot(joint.offset)
            r[joint_index] = mat2euler(M_parent)
            return index_offset

        if joint.rotation_animated():
            M_rotation, index_offset = self._extract_rotation(frame_pose, index_offset, joint)
        else:
            M_rotation = np.eye(3)

        M = M_parent.dot(M_rotation)
        position = p_parent + M_parent.dot(joint.offset) + offset_position

        rotation = np.rad2deg(mat2euler(M))
        joint_index = list(self.joints.values()).index(joint)
        p[joint_index] = position
        r[joint_index] = rotation

        for c in joint.children:
            index_offset = self._recursive_apply_frame(c, frame_pose, index_offset, p, r, M, position)

        return index_offset

    def frame_pose(self, frame):
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 3))
        frame_pose = self.keyframes[frame]
        M_parent = np.zeros((3, 3))
        M_parent[0, 0] = 1
        M_parent[1, 1] = 1
        M_parent[2, 2] = 1
        self._recursive_apply_frame(self.root, frame_pose, 0, p, r, M_parent, np.zeros(3))

        return p, r

    def all_frame_poses(self):
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 3))

        for frame in range(len(self.keyframes)):
            p[frame], r[frame] = self.frame_pose(frame)

        return p, r

    def _plot_pose(self, p, r, fig=None, ax=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')

        ax.cla()
        limits = centering(p)
#         #debug changes
#         cut_index = 3
#         ax.scatter(p[:cut_index, 0], p[:cut_index, 2], p[:cut_index, 1], c='red')
#         ax.scatter(p[cut_index:, 0], p[cut_index:, 2], p[cut_index:, 1])
        ax.scatter(p[:, 0], p[:, 2], p[:, 1])
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[2])
        ax.set_zlim(*limits[1])

        plt.draw()
        plt.pause(0.001)

    def plot_frame(self, frame, fig=None, ax=None):
        p, r = self.frame_pose(frame)
        self._plot_pose(p, r, fig, ax)

    def joint_names(self):
        return self.joints.keys()

    def parse_file(self, path):
        with open(path, 'r') as f:
            self.parse_string(f.read())

    def plot_all_frames(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.frames):
            self.plot_frame(i, fig, ax)
    
    def write_to_bvh(self, file_name):
        if self.keyframes is None:
            raise ValueError('bvh object is empty for now!')
        output = open(file_name, 'w')
        output.write('HIERARCHY\n')
        _recursive_write_hierarchy(output, self.root, 0)
        output.write(f'MOTION\n')
        output.write(f'Frames: {self.frames}\n')
        output.write(f'Frame Time: {1/self.fps:f}\n')
        for raw in self.keyframes:
            output.write(' '.join([f'{el:f}' for el in raw.reshape(-1)]))
            output.write('\n')
        print(f'Wrote "{file_name}"')
        output.close()

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"
    
    
    def get_frame_channels_recursively(self, p, joint, frame_number, parent_global_matrix, frame_channel, joint_name_index):
        if joint == self.root:
            frame_channel.extend(p[frame_number, joint_name_index[joint.name]] - joint.offset)
        if joint.name.endswith('_end'):
            return
        global_matrix = get_matrix_rotation(joint, frame_number, p, joint_name_index)
        offset_matrix = parent_global_matrix.T @ global_matrix
        angles = Rotation.from_matrix(offset_matrix).as_euler('xyz', degrees=True)[::-1]
        frame_channel.extend(angles)

        for child in joint.children:
            self.get_frame_channels_recursively(p, child, frame_number, global_matrix, frame_channel, joint_name_index)

            
            
    def get_channels(self, p, joint_name_index):
        channels = []
        for frame_number in range(self.frames):
            frame_channel = []
            self.get_frame_channels_recursively(p, self.root, frame_number, np.eye(3), frame_channel, joint_name_index)
            channels.append(frame_channel.copy())
        self.keyframes = np.array(channels)
        

        
        
    def extract_from_3D(self, p, parent_name_dict, joint_name_index, FPS, init_offsets):
        names = list(parent_name_dict.keys())

        joints_list = {}

        for nm in names:
            prnt = parent_name_dict[nm]
            joints_list[nm] = BvhJoint(nm, prnt)

        self.joints = joints_list

        for nm, joint in self.joints.items():
            for ifch in names:
                if parent_name_dict[ifch] == nm:
                    joint.add_child(self.joints[ifch])

        self.root = self.joints[[k for k,v in parent_name_dict.items() if v is None][0]]
        self.fps = FPS
        self.frames = len(p)
        # записываем названия каналов
        total_channels = 0
        for joint in self.joints.values():
            if joint == self.root:
                joint.channels = ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Yrotation', 'Xrotation']
                total_channels += 6
            elif joint.name.endswith('_end'):
                joint.channels = []
            else:
                joint.channels = ['Zrotation', 'Yrotation', 'Xrotation']
                total_channels += 3

        # запоминаем initial offsets
#         if Tpose is None:
#             Tpose = 'default_bvh_parameters/lafan_init_offsets.pickle'
#         with open(Tpose, 'rb') as f:
#             init_offsets = pickle.load(f)
        for joint in self.joints.values():
            joint.offset = init_offsets[joint.name]


        self.get_channels(p, joint_name_index)

if __name__ == '__main__':
    # create Bvh parser
    anim = Bvh()
    # parser file
    anim.parse_file("example.bvh")

    # draw the skeleton in T-pose
    anim.plot_hierarchy()

    # extract single frame pose: axis0=joint, axis1=positionXYZ/rotationXYZ
    p, r = anim.frame_pose(0)

    # extract all poses: axis0=frame, axis1=joint, axis2=positionXYZ/rotationXYZ
    all_p, all_r = anim.all_frame_poses()

    # print all joints, their positions and orientations
    for _p, _r, _j in zip(p, r, anim.joint_names()):
        print(f"{_j}: p={_p}, r={_r}")

    # draw the skeleton for the given frame
    anim.plot_frame(22)

    # show full animation
    anim.plot_all_frames()
