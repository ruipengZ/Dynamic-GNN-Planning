from collections import OrderedDict

import torch
import numpy as np
import time
import argparse
import importlib

from torch_geometric.data import Data
from configs.config import set_random_seed
from tqdm import tqdm as tqdm
from torch_sparse import coalesce
from torch_geometric.nn import knn_graph

from utils import make_gif

from result import show_result
from model import GNNet, TemporalEncoder, PolicyHead
from configs.config import set_random_seed, load_config
from utils import load_dataset, to_np

parser = argparse.ArgumentParser(description='GNN-Dynamic')
parser.add_argument('--yaml_file', type=str, default='configs/2arms.yaml',
                    help='yaml file name')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Planner():
    def __init__(self, cfg):
        #### Load environment ####
        Env = importlib.import_module('environment.' + cfg['env']['env_name'] + '_env')
        self.env = Env.Env(cfg)
        self.dof = cfg['env']['config_dim']
        self.obs_size = cfg['env']['workspace_dim']

        #### Load files ####
        self.test_obs_file = cfg['data']['testing_files_obs']
        self.test_graph_file = cfg['data']['testing_files_graph']
        self.graphs, self.obs_setting = load_dataset(self.test_graph_file, self.test_obs_file)
        self.num_graphs = len(self.graphs)

        #### Load models ####
        self.embed_size = cfg['model']['embed_size']
        self.loop = cfg['model']['loop']
        self.half_win_len = cfg['model']['window_length']
        self.model_gnn = GNNet(config_size=self.dof, embed_size=self.embed_size, obs_size=self.obs_size,
                               use_obstacles=True).to(device)
        self.model_head = PolicyHead(embed_size=self.embed_size).to(device)
        self.te_gnn = TemporalEncoder(embed_size=self.embed_size).to(device)
        self.te_head = TemporalEncoder(embed_size=self.obs_size).to(device)

        model_gnn_path = cfg['test']['saved_model_gnn_path']
        model_head_path = cfg['test']['saved_model_head_path']
        self.model_gnn.load_state_dict(torch.load(model_gnn_path, map_location=device))
        self.model_head.load_state_dict(torch.load(model_head_path, map_location=device))

        ### Explore mode: w./w.o. backtracking ###
        backtrack = cfg['test']['backtracking']
        if backtrack:
            self.search_type = 'backtrack'
            self.explore_func = self.explore_backtrack
            self.bt_buffer_size = cfg['test']['backtracking_buffer_size']
        else:
            self.search_type = 'vanilla'
            self.explore_func = self.explore_vanilla
        self.max_num_samples = cfg['test']['max_num_samples']

    @staticmethod
    def create_data(points, edge_index=None, k=50):
        goal_index = -1
        data = Data(goal=torch.FloatTensor(points[goal_index]))
        data.v = torch.FloatTensor(points)

        if edge_index is not None:
            data.edge_index = torch.tensor(edge_index.T).to(device)
        else:
            # k1 = int(np.ceil(k * np.log(len(points)) / np.log(100)))
            edge_index = knn_graph(torch.FloatTensor(data.v), k=k, loop=True)
            edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
            ### bi-directional graph
            data.edge_index, _ = coalesce(edge_index, None, len(data.v), len(data.v))

        # create labels
        labels = torch.zeros(len(data.v), 1)
        labels[goal_index, 0] = 1
        data.labels = labels

        return data

    @torch.no_grad()
    def explore_vanilla(self, graph_points, graph_edge_index, n_sample=200, t_max=2000, k=50, loop=5):
        c0 = self.env.collision_check_count
        success = False
        path = []

        ## Construct a graph ####
        points = graph_points
        data = self.create_data(points, edge_index=graph_edge_index, k=k)

        ######### Try multiple times ########
        while not success and (len(points) - 2) <= t_max:

            x = torch.arange(len(self.env.obs_points))
            pe = self.te_gnn(x).to(device)

            edge_feat, node_feat = self.model_gnn(**data.to(device).to_dict(),
                                                  pos_enc=pe,
                                                  obstacles=torch.FloatTensor(self.env.obs_points).to(device),
                                                  loop=loop)

            # explore using the head network
            cur_node = 0
            prev_node = -1
            time_tick = 0

            costs = {0: 0.}
            path = [(0, 0)]  # (node, time_cost)

            success = False
            stay_counter = 0
            ######### explore the graph ########
            while success == False:

                nonzero_indices = torch.where(edge_feat[cur_node, :, :] != 0)[0].unique()
                if nonzero_indices.size()[0] == 0:
                    print('hello')
                edges = edge_feat[cur_node, nonzero_indices, :]

                time_window = list(range(-self.half_win_len, self.half_win_len + 1))
                offsets = torch.LongTensor(time_window)
                time_window = torch.clip(int(time_tick) + offsets, 0, self.env.obs_points.shape[0] - 1)
                obs = torch.FloatTensor(self.env.obs_points[time_window].flatten())[None, :].repeat(edges.shape[0],
                                                                                                    1).to(
                    device)
                pos_enc_tw = self.te_head(time_window).flatten()[None, :].repeat(edges.shape[0], 1).to(device)

                policy = self.model_head(edges, obs, pos_enc_tw).flatten()  # [N, 32*4]
                policy = policy.cpu()

                ######### Take one step based on the policy ########
                mask = torch.arange(len(policy)).tolist()
                candidates = nonzero_indices.tolist()

                success_one_step = False

                while len(mask) != 0:  # select non-collision edge with priority

                    idx = mask[policy[mask].argmax()]
                    next_node = candidates[idx]

                    if next_node == prev_node:
                        mask.remove(idx)
                        continue

                    ############ Take the step #########
                    if self.env._edge_fp(to_np(data.v[cur_node]), to_np(data.v[next_node]), time_tick):
                        # step forward
                        success_one_step = True

                        dist = np.linalg.norm(to_np(data.v[next_node]) - to_np(data.v[cur_node]))
                        if dist == 0:  # stay
                            if stay_counter > 0:
                                mask.remove(idx)
                                success_one_step = False
                                continue
                            stay_counter += 1
                            costs[next_node] = costs[cur_node] + self.env.speed
                            time_tick += 1
                            path.append((next_node, time_tick))
                        else:
                            costs[next_node] = costs[cur_node] + dist
                            time_tick += int(np.ceil(dist / self.env.speed))
                            path.append((next_node, time_tick))
                            edge_feat[:, cur_node, :] = 0
                            stay_counter = 0

                        # update node
                        prev_node = cur_node
                        cur_node = next_node
                        break
                    ############ Search for another feasible edge #########
                    else:
                        mask.remove(idx)

                if success_one_step == False:
                    success = False
                    break

                elif self.env.in_goal_region(to_np(data.v[cur_node])):
                    check_collision = self.env.collision_check_count - c0
                    success = True

                    break

            if not success:
                new_points = self.env.uniform_sample_mine(n_sample)
                original_points = data.v.cpu()
                points = torch.cat(
                    (original_points[:-1, :], torch.FloatTensor(new_points), original_points[[-1], :]), dim=0)
                data.v = points

                edge_index = knn_graph(torch.FloatTensor(data.v), k=k, loop=True)
                edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1).to(device)
                data.edge_index, _ = coalesce(edge_index, None, len(data.v), len(data.v))

                # create labels
                labels = torch.zeros(len(data.v), 1).to(device)
                labels[-1, 0] = 1
                data.labels = labels

        if not success:
            check_collision = self.env.collision_check_count - c0

        return path, success, check_collision, np.array(data.v.cpu())


    @torch.no_grad()
    def explore_backtrack(self, graph_points, graph_edge_index, n_sample=200, t_max=2000, k=50, loop=5):
        c0 = self.env.collision_check_count
        success = False
        path = []

        ## Construct a graph ####
        points = graph_points
        data = self.create_data(points, edge_index=graph_edge_index, k=k)

        ######### Try multiple times ########
        while not success and (len(points) - 2) <= t_max:

            ## get positional encoding
            x = torch.arange(len(self.env.obs_points))
            pe = self.te_gnn(x).to(device)

            edge_feat, node_feat = self.model_gnn(**data.to(device).to_dict(),
                                                  pos_enc=pe,
                                                  obstacles=torch.FloatTensor(self.env.obs_points).to(device),
                                                  loop=loop)

            # explore using the head network
            cur_node = 0
            prev_node = -1
            time_tick = 0

            costs = {0: 0.}
            path = [(0, 0)]  # (node, time_cost)

            prevs = OrderedDict()
            prevs[(0, 0)] = (-1, -1)

            success = False
            stay_counter = 0
            history_stack = []
            back_tracking = False
            ######### explore the graph ########
            while success == False:

                nonzero_indices = torch.where(edge_feat[cur_node, :, :] != 0)[0].unique()
                edges = edge_feat[cur_node, nonzero_indices, :]

                time_window = list(range(-self.half_win_len, self.half_win_len + 1))
                offsets = torch.LongTensor(time_window)
                time_window = torch.clip(int(time_tick) + offsets, 0, self.env.obs_points.shape[0] - 1)
                obs = torch.FloatTensor(self.env.obs_points[time_window].flatten())[None, :].repeat(edges.shape[0], 1).to(
                    device)
                pos_enc_tw = self.te_head(time_window).flatten()[None, :].repeat(edges.shape[0], 1).to(device)

                policy = self.model_head(edges, obs, pos_enc_tw).flatten()

                if back_tracking == False:
                    children = nonzero_indices[torch.argsort(policy, descending=True)].tolist()
                    children.reverse()
                    cur_stack = [(cur_node, child, time_tick) for child in children]
                else:
                    cur_stack = history_stack

                ######### Take one step based on the policy ########

                success_one_step = False

                while len(cur_stack) != 0:  # select non-collision edge with priority

                    cur_node, next_node, time_tick = cur_stack.pop()

                    if next_node == prev_node:
                        continue

                    ############ Take the step #########
                    if self.env._edge_fp(to_np(data.v[cur_node]), to_np(data.v[next_node]), time_tick):
                        # step forward
                        success_one_step = True
                        ## put the following k-1 children into the history stack
                        history_stack.extend([cur_stack.pop() for _ in range(min(self.bt_buffer_size-1, len(cur_stack)))])
                        ## end backtracking mode
                        back_tracking = False

                        dist = np.linalg.norm(to_np(data.v[next_node]) - to_np(data.v[cur_node]))
                        if dist == 0:  # stay
                            stay_counter += 1
                            costs[next_node] = costs[cur_node] + self.env.speed
                            old_time = time_tick
                            time_tick += 1
                            prevs[(next_node, time_tick)] = (cur_node, old_time)

                        else:
                            costs[next_node] = costs[cur_node] + dist
                            old_time = time_tick
                            time_tick += int(np.ceil(dist / self.env.speed))
                            prevs[(next_node, time_tick)] = (cur_node, old_time)
                            edge_feat[:, cur_node, :] = 0
                            stay_counter = 0

                        # update node
                        prev_node = cur_node
                        cur_node = next_node
                        break
                    ############ Search for another feasible edge #########
                    else:
                        continue

                if success_one_step == False:
                    if len(history_stack) == 0:
                        success = False
                        break
                    ### start backtracking
                    back_tracking = True
                    cur_node, next_node, time_tick = history_stack.pop()
                    continue

                elif self.env.in_goal_region(to_np(data.v[cur_node])):
                    check_collision = self.env.collision_check_count - c0

                    success = True
                    p_time = time_tick
                    p_node = cur_node
                    path = [(p_node, p_time)]
                    while p_node != 0:
                        path.append(prevs[(p_node, p_time)])
                        p_node, p_time = prevs[(p_node, p_time)]
                    path.reverse()

                    break

            if not success:
                new_points = self.env.uniform_sample_mine(n_sample)
                original_points = data.v.cpu()
                points = torch.cat(
                    (original_points[:-1, :], torch.FloatTensor(new_points), original_points[[-1], :]), dim=0)
                data.v = points

                edge_index = knn_graph(torch.FloatTensor(data.v), k=k, loop=True)
                edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1).to(device)
                data.edge_index, _ = coalesce(edge_index, None, len(data.v), len(data.v))

                # create labels
                labels = torch.zeros(len(data.v), 1).to(device)
                labels[-1, 0] = 1
                data.labels = labels

        if not success:
            check_collision = self.env.collision_check_count - c0

        return path, success, check_collision, np.array(data.v.cpu())


    def motion_planning(self, seed, indexes, use_tqdm=False, t_max=2000, k=50, **kwargs):

        set_random_seed(seed)
        self.model_gnn.eval()
        self.model_head.eval()

        result_dict = {'success': [], 'path_time': [], 'path': [], 'check_collision': [], 'points': [],
                       'inference_time': []}

        pbar = tqdm(indexes) if use_tqdm else indexes
        for index in pbar:
            self.env.init_new_problem(index=index, setting_dict=self.obs_setting)
            graph_points = self.graphs[index][0]
            graph_edge_index = self.graphs[index][1]

            t0 = time.time()
            result = self.explore_func(graph_points, graph_edge_index, t_max=t_max, k=k)

            result_dict['inference_time'].append(time.time() - t0)

            path, success, check_collision, points = result
            result_dict['points'].append(points)
            result_dict['success'].append(success)
            result_dict['path'].append(path)
            result_dict['path_time'].append(path[-1][-1])
            result_dict['check_collision'].append(check_collision)

        return result_dict


if __name__ == '__main__':
    cfg = load_config(args.yaml_file)
    model_name = 'arm2'

    planner = Planner(cfg)
    indexes = range(planner.num_graphs)
    result_dict = planner.motion_planning(seed=1234, indexes=indexes, use_tqdm=True, t_max=planner.max_num_samples)
    show_result(model_name=model_name, gt_file=planner.test_graph_file, result_dict=result_dict, index_list=indexes)
    np.savez(f'result/{model_name}_{planner.search_type}.npz', **result_dict)
