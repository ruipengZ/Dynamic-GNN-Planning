from PIL import Image
import numpy as np
import pickle
import torch

def make_gif(gifs, path, duration=50, loop=0):
    a_frames = []
    for im_frame in gifs:
        a_frames.append(np.asarray(im_frame))
    a = np.stack(a_frames)

    ims = [Image.fromarray(a_frame) for a_frame in a]
    ims[0].save(path, save_all=True, append_images=ims[1:], loop=loop, duration=duration)
    return

def make_png(gifs, path, duration=50, loop=0):
    a_frames = []
    for im_frame in gifs:
        a_frames.append(np.asarray(im_frame))
    a = np.stack(a_frames)

    ims = [Image.fromarray(a_frame) for a_frame in a]
    for i in range(len(ims)):
        ims[i].save(f'{path[:-4]}_{i}.png')
    return

def to_np(tensor):
    return tensor.data.cpu().numpy()


def load_dataset(data_path_graph, data_path_obs):
    graphs = []

    init_states_list = []
    goal_states_list = []
    obs_pos_list = []
    obs_ori_list = []
    obs_traj_list = []
    obs_setting = {}

    ### load testcase and SIPP result ####
    for file_idx in range(len(data_path_graph)):
        with np.load(data_path_obs[file_idx]) as f:
            init_states_list.append(f['init_states'])
            goal_states_list.append(f['goal_states'])
            obs_pos_list.append(f['obs_pos'])
            obs_ori_list.append(f['obs_ori'])
            obs_traj_list.append(f['obs_traj'])

        print(f'{data_path_obs[file_idx]} loaded')

        with open(data_path_graph[file_idx], 'rb') as f:
            load_graph = pickle.load(f)
            graphs.extend(load_graph)
        print(f'{data_path_graph[file_idx]} loaded')

    obs_setting['init_states'] = np.concatenate(init_states_list)
    obs_setting['goal_states'] = np.concatenate(goal_states_list)
    obs_setting['obs_pos'] = np.concatenate(obs_pos_list)
    obs_setting['obs_ori'] = np.concatenate(obs_ori_list)
    obs_setting['obs_traj'] = np.concatenate(obs_traj_list)

    return graphs, obs_setting

def load_path_time(input_path_time, device):
    gt_path = [x[0] for x in input_path_time]
    gt_time = [x[1] for x in input_path_time]

    path_time = torch.zeros((0, 2), device=device)
    for i in range(len(gt_path)):
        if i > 0 and gt_path[i] == gt_path[i - 1]:
            for j in range(gt_time[i - 1], gt_time[i] - 1):
                new_path_time = torch.tensor([[gt_path[i], j + 1]], device=device)
                path_time = torch.cat([path_time, new_path_time], axis=0)

        new_path_time = torch.tensor([[gt_path[i], gt_time[i]]], device=device)
        path_time = torch.cat([path_time, new_path_time], axis=0)

    path_time = path_time.int()

    return path_time
