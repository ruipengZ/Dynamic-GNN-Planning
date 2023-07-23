import argparse
import importlib

from tqdm import tqdm
import pickle

import numpy as np
from configs.config import load_config
from sipp import SIPP, construct_graph

parser = argparse.ArgumentParser(description='GNN-Dynamic')
parser.add_argument('--yaml_file', type=str, default='configs/2arms.yaml',
                    help='yaml file name')
args = parser.parse_args()
cfg = load_config(args.yaml_file)

valid_list_file_path = f"testcase/{cfg['env']['env_name']}/feasible_list.npy"
graph_file_path = cfg['data']['training_files_graph'][0]
obs_file_path = cfg['data']['training_files_obs'][0]


class CaseGenerator():
    def __init__(self, cfg, alg, n_points):
        self.alg = alg(cfg)
        self.env = self.alg.env

        self.traj_obs_timestep = cfg['env']['traj_obs_timestep']
        self.length = (self.traj_obs_timestep - 1) * 1/(cfg['env']['unit_timestep'] - 1)
        self.n_points = n_points



    def resetObs(self):
        while True:
            radius = np.random.random()
            angle = np.random.random() * 2 * np.pi
            base_x = np.random.random() + 0.5 + radius*np.cos(angle)
            base_y = radius * np.sin(angle)
            if base_y > 0.5:
                break

        theta = 3/2 *np.pi + angle

        self.env.resetBaseOrientation([base_x, base_y, 0], [0, 0, np.sin(theta/2), np.cos(theta/2)])
        self.obs_pos = [base_x, base_y, 0]
        self.obs_ori = [0, 0, np.sin(theta/2), np.cos(theta/2)]


    def genTrajObs(self):
        source = np.random.uniform([0.] * 2, [np.pi / 2] * 2)
        goal = np.random.uniform([0.] * 2, [np.pi] * 2)
        goal = source + self.length * (goal - source) / np.linalg.norm(goal - source)
        ts = np.linspace(0, 1., self.traj_obs_timestep)
        self.obs_traj = source.reshape((1, -1)) + ts.reshape((-1, 1)) * ((goal - source).reshape((1, -1)))
        return self.obs_traj

    def genSamples(self):
        source = np.random.uniform([0.] * 2, [np.pi/2] * 2)
        goal = np.random.uniform([0.] * 2, [np.pi] * 2)
        goal = source + self.length * (goal - source) / np.linalg.norm(goal-source)


        points = self.env.uniform_sample_mine(n=self.n_points)
        points = np.insert(points, 0, source, axis=0)
        points = np.append(points, goal.reshape(1, -1), axis=0)
        return points

    def algorithm(self):
        self.resetObs()
        obs_traj = self.genTrajObs()
        while True:
            points = self.genSamples()
            edges, edge_cost, edge_index = construct_graph(points)
            output = self.alg.path_planning(obs_traj, points, edges, edge_cost)

            if output[0] == False:
                continue

            arr_time, prev, safe_intervals, path_time, feasible, end_time, collision_check = output

            return points, edge_index, arr_time, prev, safe_intervals, path_time, feasible, end_time, collision_check




def main(pid, input_epochs):
    obs_pos = []
    obs_ori = []
    obs_traj = []
    data = []
    init_states = []
    goal_states = []

    tqdm_text = "#" + "{}".format(pid).zfill(3)
    low = input_epochs[0]
    high = input_epochs[1]
    epochs = range(low, high)

    feasible_list = []

    with tqdm(total=high-low, desc=tqdm_text, position=pid + 1) as pbar:

        for problem_index in epochs:

            np.random.seed(problem_index)
            gen = CaseGenerator(cfg, SIPP, n_points=1000)
            output = gen.algorithm()

            if output == False:
                print(f'{problem_index} infeasible')
                continue
            else:
                points, edge_index, arr_time, prev, safe_intervals, path_time, feasible, end_time, collision_check = output
                print(f'{problem_index} success:', collision_check)

            feasible_list.append(problem_index)

            obs_pos.append(gen.obs_pos)
            obs_ori.append(gen.obs_ori)
            obs_traj.append(gen.obs_traj)
            init_states.append(points[0])
            goal_states.append(points[len(points) - 1])

            data.append(
                (points, edge_index, arr_time, prev, safe_intervals, path_time, feasible, end_time, collision_check))


            pbar.update(1)

    return data, obs_pos, obs_ori, obs_traj, init_states, goal_states, feasible_list



if __name__ == '__main__':
    import multiprocessing as mp
    num_cores = int(mp.cpu_count())
    print("# cores: " + str(num_cores))

    num_cores = int(mp.cpu_count())
    print("# cores: " + str(num_cores))

    param_list = []
    total_num_testcases = 100
    each_num = total_num_testcases // num_cores

    start = 0
    for i in range(num_cores):
        param_list.append((start, start + each_num))
        start += each_num



    with mp.Pool(processes=num_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        results = [pool.apply_async(main, args=(i, epoch,)) for i, epoch in enumerate(param_list)]
        results = [p.get() for p in results]

    print("\n" * (len(param_list) + 1))
    print('finish all sipp')

    data = []
    obs_pos_all = []
    obs_ori_all = []
    obs_traj_all = []
    init_states_all = []
    goal_states_all = []
    feasible_list = []
    for i in range(len(results)):
        data.extend(results[i][0])
        obs_pos_all.extend(results[i][1])
        obs_ori_all.extend(results[i][2])
        obs_traj_all.extend(results[i][3])
        init_states_all.extend(results[i][4])
        goal_states_all.extend(results[i][5])
        feasible_list.extend(results[i][6])

    np.save(valid_list_file_path, feasible_list)
    print(len(feasible_list))

    with open(graph_file_path, 'wb') as f:
        pickle.dump(data, f, pickle.DEFAULT_PROTOCOL)

    a = {'obs_pos': obs_pos_all,
         'obs_ori': obs_ori_all,
         'obs_traj': obs_traj_all,
         'init_states': init_states_all,
         'goal_states': goal_states_all
         }
    np.savez(obs_file_path, **a)

    print(f'********* Saving {total_num_testcases} test files ************')


