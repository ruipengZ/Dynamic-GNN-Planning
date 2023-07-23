import time
import argparse

from tqdm import tqdm
import pickle
import pybullet as p
import numpy as np
from configs.config import load_config
from sipp import SIPP, construct_graph

parser = argparse.ArgumentParser(description='GNN-Dynamic')
parser.add_argument('--yaml_file', type=str, default='configs/kuka.yaml',
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
            radius = np.random.random() * 0.05 + 0.5
            angle = np.random.random() * np.pi - np.pi / 2
            base_x = radius * np.cos(angle)
            base_y = radius * np.sin(angle)
            if base_y > 0.01:
                break

        theta = -3 / 2 * np.pi + angle
        self.theta = theta

        self.env.resetBaseOrientation([base_x, base_y, 0], [0, 0, np.sin(self.theta / 2), np.cos(self.theta / 2)])
        self.env.obs_pos = [base_x, base_y, 0]
        self.env.obs_ori = [0, 0, np.sin(self.theta / 2), np.cos(self.theta / 2)]
        self.obs_pos = [base_x, base_y, 0]
        self.obs_ori = [0, 0, np.sin(self.theta / 2), np.cos(self.theta / 2)]
        self.base_x = base_x
        self.base_y = base_y

    def genSourceAll(self):
        dest_x_obs = np.random.random() * self.base_x / 4 + self.base_x / 2
        dest_y_obs = self.base_x / self.base_y * (dest_x_obs - self.base_x / 2) + self.base_y / 2

        dest_z_obs = np.random.choice([1, -1, 0.5, -0.5])
        dest_source_obs = [dest_x_obs, dest_y_obs, dest_z_obs]
        obs_source = p.calculateInverseKinematics(self.env.stick2, 6, dest_source_obs)

        count = 0
        while True:
            if count > 100:
                return False
            dest_x_mine = dest_x_obs + np.random.uniform(-0.1, 0.1)
            dest_y_mine = dest_y_obs + np.random.uniform(-0.1, 0.1)
            dest_z_mine = dest_z_obs + np.random.uniform(-0.1, 0.1)
            dest_source_mine = [dest_x_mine, dest_y_mine, dest_z_mine]

            mine_source = p.calculateInverseKinematics(self.env.stick1, 6, dest_source_mine)

            self.env.set_config_mine(mine_source)
            self.env.set_config_obs(obs_source)
            # p.stepSimulation()

            if self.env.check_collision():
                break
            count += 1

        self.obs_source = np.array(obs_source)
        self.mine_source = np.array(mine_source)
        return True

    def genTrajObs(self):
        self.env.set_config_obs(self.obs_source)
        e_source = p.getLinkState(self.env.stick2, 6)[0]

        if e_source[2] < 0.3:
            ### go up
            destination = [e_source[0], e_source[1], 1]
        else:
            ### go down
            destination = [e_source[0], e_source[1], -1]

        self.env.set_config_obs(destination)
        # p.stepSimulation()
        obs_goal = p.calculateInverseKinematics(self.env.stick2, 6, destination)
        obs_goal += np.random.uniform([-0.01] * self.env.config_dim, [0.01] * self.env.config_dim)


        obs_goal = self.obs_source + self.length * (obs_goal - self.obs_source) / np.linalg.norm(
            obs_goal - self.obs_source)
        ts = np.linspace(0, 1., self.traj_obs_timestep)
        self.obs_traj = self.obs_source.reshape((1, -1)) + ts.reshape((-1, 1)) * (
            (obs_goal - self.obs_source).reshape((1, -1)))

        return self.obs_traj

    def genGoalMine(self):

        self.env.set_config_mine(self.mine_source)
        e_source = p.getLinkState(self.env.stick1, 6)[0]

        if e_source[2] < 0.4:
            ### go up
            destination = [e_source[0], e_source[1], 1]
        else:
            ### go down
            destination = [e_source[0], e_source[1], -1]

        count = 0
        while True:
            if count > 100:
                return False
            dest_x_mine = destination[0] + np.random.uniform(-0.2, 0.2)
            dest_y_mine = destination[1] + np.random.uniform(-0.2, 0.2)
            dest_z_mine = destination[2] + np.random.uniform(-0.2, 0.2)
            destination = [dest_x_mine, dest_y_mine, dest_z_mine]

            mine_goal = p.calculateInverseKinematics(self.env.stick1, 6, destination)

            self.mine_goal = self.mine_source + self.length * (mine_goal - self.mine_source) / np.linalg.norm(
                mine_goal - self.mine_source)

            self.env.set_config_mine(self.mine_goal)
            self.env.set_config_obs(self.obs_traj[-1])
            if self.env.check_collision():
                break
            count += 1
        return True

    def genVoxels(self):

        halfExtents_list = []
        basePosition_list = []
        num_voxels = np.random.choice([2, 3, 4])
        for i in range(num_voxels):
            count = 0
            while True:
                if count > 100:
                    return False, False
                halfExtents = np.random.uniform([.1] * 3, [.2] * 3)
                basePosition = np.random.uniform([-.5] * 3, [.6] * 3)
                body_id = self.env.create_voxel(halfExtents, basePosition)

                ## set to init position ###
                self.env.set_config_mine(self.mine_source)
                self.env.set_config_obs(self.obs_source)

                if not self.env.check_collision_body(body_id):
                    p.removeBody(body_id)
                    count += 1
                    continue

                self.env.set_config_mine(self.mine_goal)
                if not self.env.check_collision_body(body_id):
                    p.removeBody(body_id)
                    count += 1
                    continue

                if not self.env.check_traj_collision_body(self.obs_traj, body_id):
                    p.removeBody(body_id)
                    continue
                else:
                    p.removeBody(body_id)
                    break


            self.env.create_voxel(halfExtents, basePosition)
            halfExtents_list.append(halfExtents)
            basePosition_list.append(basePosition)

        return halfExtents_list, basePosition_list

    def genSamples(self):
        source = self.mine_source
        goal = self.mine_goal
        points = self.env.uniform_sample_mine_wo_collision(n=self.n_points)
        points = np.insert(points, 0, source, axis=0)
        points = np.append(points, goal.reshape(1, -1), axis=0)
        return points

    def algorithm(self):
        self.resetObs()
        source_result = self.genSourceAll()
        if source_result == False:
            return False
        obs_traj = self.genTrajObs()
        goal_result = self.genGoalMine()
        if goal_result == False:
            return False
        halfExtents_list, basePosition_list = self.genVoxels()
        if halfExtents_list == False:
            return False

        while True:
            points = self.genSamples()

            edges, edge_cost, edge_index = construct_graph(points)
            t0 = time.perf_counter()

            output = self.alg.path_planning(obs_traj, points, edges, edge_cost)
            plan_time = time.perf_counter()-t0

            if output[0] == False :
                return False

            arr_time, prev, safe_intervals, path, feasible, end_time, collision_check = output

            t1 = time.time()

            return points, edges, edge_cost, edge_index, halfExtents_list, basePosition_list, arr_time, prev, safe_intervals, path, plan_time, end_time, collision_check



def main(pid, input_epochs):
    obs_pos = []
    obs_ori = []
    obs_traj = []
    data = []
    init_states = []
    goal_states = []


    low = input_epochs[0]
    high = input_epochs[1]
    tqdm_text = "#" + "{}".format(pid).zfill(3)
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
                points, neighbors, edge_cost, edge_index, halfExtents_list, basePosition_list, arr_time, prev, safe_intervals, path, plan_time, end_time, collision_check = output
                print(f'{problem_index} success:', collision_check)

            feasible_list.append(problem_index)

            obs_pos.append(gen.obs_pos)
            obs_ori.append(gen.obs_ori)
            obs_traj.append(gen.obs_traj)
            init_states.append(points[0])
            goal_states.append(points[len(points)-1])

            data.append((points, edge_index, halfExtents_list, basePosition_list, arr_time, prev, safe_intervals, path, plan_time, end_time, collision_check))


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


