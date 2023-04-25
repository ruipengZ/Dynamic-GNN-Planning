import sys
sys.path.append('..')

import argparse
import importlib

from collections import OrderedDict
import numpy as np
import torch
import pybullet as p
from torch_geometric.nn import knn_graph
from collections import defaultdict
from torch_sparse import coalesce
from utils import make_gif
import time

from configs.config import load_config


INFINITY = float('inf')

parser = argparse.ArgumentParser(description='GNN-Dynamic')
parser.add_argument('--yaml_file', type=str, default='configs/2arms.yaml',
                    help='yaml file name')
args = parser.parse_args()

class SIPP:
    def __init__(self, cfg):
        Env = importlib.import_module('environment.' + cfg['env']['env_name'] + '_env')
        self.env = Env.Env(cfg)

        self.source_idx = 0
        self.k = cfg['env']['unit_timestep']
        self.collision_check = 0
        self.RRT_EPS = cfg['env']['RRT_EPS']

    def get_cfg_obs(self):
        self.speed = 1. / (self.k - 1)
        time_cfg_obs = OrderedDict()
        max_time_obs = len(self.obs_traj)-1
        for i in range(max_time_obs+1):
            time_cfg_obs[i] = self.obs_traj[i]
        return max_time_obs, time_cfg_obs


    def get_safe_intervals(self):
        # safe_intervals {node:[(startTime, endTime), (startTime, endTime),...]}
        safe_intervals = {}
        cc = 0
        for mine in range(len(self.points)):
            c_mine = self.points[mine]
            self.env.set_config_mine(c_mine)
            interval = []
            prev_t = 0
            for time in self.time_cfg_obs:

                c_obs = self.time_cfg_obs[time]

                self.env.set_config_obs(c_obs)
                cc += 1
                if not self.env.check_collision(): # collision
                    if time > prev_t:
                        interval.append((prev_t, time-1))
                    prev_t = time + 1

                elif time==next(reversed(self.time_cfg_obs)):
                    interval.append((prev_t, INFINITY))

            safe_intervals[mine] = interval
        return safe_intervals

    @staticmethod
    def min_dist(q, dist):
        """
        Returns the node with the smallest distance in q.
        Implemented to keep the main algorithm clean.
        """
        min_node = None
        for node in q:
            if min_node is None:
                min_node = node
            elif dist[node] < dist[min_node]:
                min_node = node

        return min_node

    def heuristic(self, nodes, edges, costs, source):
        """
        dijkstra search in configuration space without collision check
        """

        q = set()
        dist = {}

        for v in nodes:  # initialization
            dist[v] = INFINITY  # unknown distance from goal to v
            q.add(v)  # all nodes initially in q (unvisited nodes)

        # distance from goal to every node
        dist[source] = 0

        while q:
            # node with the least distance selected first
            u = self.min_dist(q, dist)

            q.remove(u)

            for index, v in enumerate(edges[u]):
                alt = dist[u] + costs[u][index]
                if alt < dist[v]:
                    # a shorter path to v has been found
                    dist[v] = alt

        return dist

    def check_edge_collision(self, src, dest, arr_t, mov_time, src_arr_time):
        mov_start_t = arr_t - mov_time
        disp_mine = self.points[dest] - self.points[src]

        d = np.linalg.norm(disp_mine)
        K = int(np.ceil(d / self.speed))
        self.collision_check += d//self.RRT_EPS

        # mine arm waiting
        self.env.set_config_mine(self.points[src])
        for time in range(src_arr_time, mov_start_t+1):
            self.collision_check += 1
            self.env.set_config_obs(self.time_cfg_obs[min(time, self.max_time_obs)])
            if not self.env.check_collision():
                return False

        # mine arm moving
        for k in range(1, K+1):
            c_mine = self.points[src] + k * 1. / K * disp_mine
            self.env.set_config_mine(c_mine)

            c_obs = self.time_cfg_obs[min(mov_start_t + k, self.max_time_obs)]
            self.env.set_config_obs(c_obs)

            if not self.env.check_collision():
                return False

        return True


    def get_no_collision_time(self, src, dest, start_t, end_t, interval, mov_time, src_arr_time):
        overlap_s = max(start_t, interval[0])
        overlap_e = min(end_t, interval[1])
        if self.max_time_obs + mov_time <= overlap_s:
            if self.check_edge_collision(src, dest, overlap_s, mov_time, src_arr_time):
                return overlap_s
            else:
                return None

        for arr_t in range(overlap_s, min(overlap_e+1, self.max_time_obs + mov_time)):
            # check collision on the edge
            if self.check_edge_collision(src, dest, arr_t, mov_time, src_arr_time):
                return arr_t

        return None

    def get_successors(self, edges, src, src_interval_idx, src_arr_time):
        # successor: (node, index of interval, latest non-collision time)
        succs = set()
        for dest in edges[src]:

            mov_time = int(np.ceil(np.linalg.norm(self.points[src] - self.points[dest]) / self.speed))
            start_t = src_arr_time + mov_time
            end_t = self.safe_intervals[src][src_interval_idx][1] + mov_time

            for idx, interval in enumerate(self.safe_intervals[dest]):
                if interval[0]>end_t or interval[1]<start_t:
                    continue
                arr_t = self.get_no_collision_time(src, dest, start_t, end_t, interval, mov_time, src_arr_time)
                if not arr_t:
                    continue
                succ = (dest, idx, arr_t)
                succs.add(succ)
        return succs

    def check_free_space(self, obs_traj, points, edges, edge_cost, source_idx=None, goal_idx=None):
        self.obs_traj = obs_traj
        self.points = points
        self.edges = edges
        self.edge_cost = edge_cost
        if source_idx:
            self.source_idx = source_idx
        else:
            self.source_idx = 0
        if goal_idx:
            if goal_idx == -1:
                self.goal_idx = len(self.points) - 1
            else:
                self.goal_idx = goal_idx
        else:
            self.goal_idx = len(self.points) - 1
        self.max_time_obs, self.time_cfg_obs = self.get_cfg_obs()
        self.safe_intervals = self.get_safe_intervals()

        if len(self.safe_intervals[self.source_idx])==0 or self.safe_intervals[self.source_idx][0][0] != 0:
            return False
        else:
            return True

    def path_planning(self, obs_traj, points, edges, edge_cost, source_idx=None, goal_idx=None):

        if self.check_free_space(obs_traj, points, edges, edge_cost, source_idx, goal_idx) == False:
            return False, 'free_space'


        nodes_idx = list(range(len(self.points)))
        h = self.heuristic(nodes_idx, self.edges, self.edge_cost, self.goal_idx)

        # node (index of node, index of interval)
        g = {}
        g[(self.source_idx, 0)] = 0
        f = {}
        f[(self.source_idx, 0)] = h[self.source_idx]
        prev = {}
        prev[(self.source_idx, 0)] = (-1, -1)

        # earliest non-collision arrival time for every node
        arr_time = {}
        arr_time[(self.source_idx, 0)] = 0

        open = set()
        open.add((self.source_idx, 0))

        t0 = time.time()
        while open:
            s = min(open, key=lambda node:f[node])
            open.remove(s)

            s_index, s_interval_idx = s
            s_arr_t = arr_time[s]
            succs = self.get_successors(self.edges, s_index, s_interval_idx, s_arr_t)
            for item in succs:
                succ_idx, succ_interval_idx, succ_arr_t = item
                succ = (succ_idx, succ_interval_idx)
                if succ not in f.keys():
                    f[succ] = INFINITY
                    g[succ] = INFINITY
                if g[succ] > succ_arr_t:
                    prev[succ] = s
                    g[succ] = succ_arr_t
                    arr_time[succ] = succ_arr_t
                    f[succ] = g[succ]+ h[succ_idx]
                    open.add(succ)


        if self.goal_idx not in [n[0] for n in arr_time.keys()]:
            feasible = False
            return False, 'infeasible'
        else:
            feasible = True
            path = self.generatePath(arr_time, prev)

        return arr_time, prev, self.safe_intervals, path, feasible, path[-1][-1], self.collision_check

    def generatePath(self, arr_time, prev):
        path = []
        for i in range(len(self.safe_intervals[self.goal_idx])):
            goal = (self.goal_idx, i)
            if goal in arr_time.keys():
                path = [(self.goal_idx, arr_time[goal])]
                break

        s = goal
        while prev[s] != (-1,-1):
            path.append((prev[s][0], arr_time[prev[s]]))
            s = prev[s]
        path.reverse()
        return path

    def plot(self, input_path, make_gif=False):
        """
        input_path: [(point_idx, arr_time)]
        """
        path = []
        time = []
        for _, item in enumerate(input_path):
            path.append(self.points[item[0]])
            time.append(item[1])
        path = np.array(path)

        self.env.set_config_obs(self.obs_traj[0])
        self.env.set_config_mine(path[0])

        gifs = []
        current_state = 0
        current_time = 0

        while True:
            mov_time = int(np.ceil(np.linalg.norm(path[current_state + 1] - path[current_state]) / self.speed))
            # mine waiting
            for t in range(current_time, time[current_state + 1] - mov_time + 1):
                self.env.set_config_obs(self.time_cfg_obs[min(t, self.max_time_obs)])
                if self.env.check_collision() == False:
                    print(t)
                    print("COLLIDE waiting!!!\n")
                if make_gif:
                    gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])

            current_time = time[current_state + 1] - mov_time

            # mine moving to the next state
            disp = path[current_state + 1] - path[current_state]
            d = np.linalg.norm(disp)
            K = int(np.ceil(d / self.speed))

            for k in range(1, K+1):
                c_mine = path[current_state] + k * 1. / K * disp
                self.env.set_config_mine(c_mine)
                current_time += 1
                self.env.set_config_obs(self.time_cfg_obs[min(current_time, self.max_time_obs)])

                if self.env.check_collision() == False:
                    print(current_time)
                    print("COLLIDE moving!!!\n")
                if make_gif:
                    gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])

            current_state += 1  # moved to the next state

            if current_state == len(path) - 1:
                break

        return gifs

def plot_sipp(env, points, sipp_path, obs_traj, make_gif=False):
    """
    sipp_path: [(point_idx, arr_time)]
    """
    path = []
    time = []
    for _, item in enumerate(sipp_path):
        path.append(points[item[0]])
        time.append(item[1])
    path = np.array(path)

    env.set_config_obs(obs_traj[0])
    env.set_config_mine(path[0])

    gifs = []
    current_state = 0
    current_time = 0

    u_d = 0.1
    time_cfg_obs = OrderedDict()
    time_cfg_obs[0] = obs_traj[0]
    max_time_obs = 0
    for i in range(len(obs_traj) - 1):
        disp = obs_traj[i + 1] - obs_traj[i]
        d = np.linalg.norm(disp)
        K = int(np.ceil(d / u_d))
        for k in range(1, K + 1):
            max_time_obs += 1
            c = obs_traj[i] + k * 1. / K * disp
            time_cfg_obs[max_time_obs] = c

    while True:
        mov_time = int(np.ceil(np.linalg.norm(path[current_state + 1] - path[current_state]) / u_d))
        # mine waiting
        for t in range(current_time, time[current_state + 1] - mov_time + 1):
            env.set_config_obs(time_cfg_obs[min(t, max_time_obs)])
            if env.check_collision() == False:
                print(t)
                print("COLLIDE waiting!!!\n")
            if make_gif:
                gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])

        current_time = time[current_state + 1] - mov_time

        # mine moving to the next state
        disp = path[current_state + 1] - path[current_state]
        d = np.linalg.norm(disp)
        K = int(np.ceil(d / u_d))

        for k in range(1, K + 1):
            c_mine = path[current_state] + k * 1. / K * disp
            env.set_config_mine(c_mine)
            current_time += 1
            env.set_config_obs(time_cfg_obs[min(current_time, max_time_obs)])

            if env.check_collision() == False:
                print(current_time)
                print("COLLIDE moving!!!\n")
            if make_gif:
                gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])

        current_state += 1  # moved to the next state

        if current_state == len(path) - 1:
            break

    return gifs



def plot_traj(env, traj_path, init_mine, end_mine, make_gif=False):

    gifs = []
    current_state = 0
    current_time = 0
    u_d = 0.01


    mov_time_mine = int(np.ceil(np.linalg.norm(end_mine - init_mine) / u_d))
    disp_mine = end_mine - init_mine
    while True:
        mov_time_obs = int(np.ceil(np.linalg.norm(traj_path[current_state + 1] - traj_path[current_state]) / u_d))

        disp_obs = traj_path[current_state + 1] - traj_path[current_state]

        for t in range(mov_time_obs):
            current_time += 1

            c_obs = traj_path[current_state] + t * 1 / mov_time_obs * disp_obs
            env.set_config_obs(c_obs)
            if current_time <= mov_time_mine:
                c_mine = init_mine + current_time * 1 / mov_time_mine * disp_mine
                env.set_config_mine(c_mine)

            if make_gif:
                gifs.append(p.getCameraImage(width=1080, height=720, lightDirection=[0, 0, -1], shadow=0,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)[2])

        current_state += 1  # moved to the next state

        if current_state == len(traj_path) - 1:
            break

    return gifs


def construct_graph(points):
    edge_index = knn_graph(torch.FloatTensor(points), k=50, loop=True)
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    edge_index_torch, _ = coalesce(edge_index, None, len(points), len(points))
    edge_index = edge_index_torch.data.cpu().numpy().T
    edge_cost = defaultdict(list)
    edges = defaultdict(list)
    for i, edge in enumerate(edge_index):
        edge_cost[edge[1]].append(np.linalg.norm(points[edge[1]] - points[edge[0]]))

        edges[edge[1]].append(edge[0])

    return edges, edge_cost, edge_index


if __name__ == '__main__':
    cfg = load_config(args.yaml_file)
    sipp = SIPP(cfg)

    #############
    path_obs_1 = np.array([np.pi / 3] * 2)
    path_obs_2 = np.array([0] * 2)
    path_obs_3 = np.array([np.pi / 3] * 2)
    obs_traj = [np.array([np.pi / 4] * 2)]

    init_mine = np.array([0.] * 2)
    end_mine = np.array([np.pi*7/8, 0])

    points = sipp.env.uniform_sample_mine(n=700)
    points = np.insert(points, 0, init_mine, axis=0)
    points = np.append(points, end_mine.reshape(1, -1), axis=0)

    edges, edge_cost, edge_index = construct_graph(points)
    output = sipp.path_planning(obs_traj, points, edges, edge_cost)
    if output != False:
        path = output[3]
        gifs = sipp.plot(path, make_gif=True)
        make_gif(gifs, 'test.gif', duration=20)

