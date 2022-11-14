import argparse
import importlib

import torch
import numpy as np
from torch_geometric.data import Data
from configs.config import set_random_seed, load_config
from tqdm import tqdm as tqdm

from model import GNNet, TemporalEncoder, PolicyHead
from utils import load_dataset, load_path_time

parser = argparse.ArgumentParser(description='GNN-Dynamic')
parser.add_argument('--yaml_file', type=str, default='configs/2arms.yaml',
                    help='yaml file name')
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, cfg):
        Env = importlib.import_module('environment.' + cfg['env']['env_name'] + '_env')
        self.env = Env.Env(cfg)

        self.dof = cfg['env']['config_dim']
        self.obs_size = cfg['env']['workspace_dim']

        self.embed_size = cfg['model']['embed_size']
        self.loop = cfg['model']['loop']
        self.half_win_len = cfg['model']['window_length']

        self.data_path_graph = cfg['data']['training_files_graph']
        self.data_path_obs = cfg['data']['training_files_obs']

        self.model_gnn_path = cfg['train']['output_model_gnn_path']
        self.model_head_path = cfg['train']['output_model_head_path']
        self.epochs = cfg['train']['epochs']
        self.lr = cfg['train']['lr']


        self.model_gnn = GNNet(config_size=self.dof, embed_size=self.embed_size, obs_size=self.obs_size, use_obstacles=True).to(device)
        self.model_head = PolicyHead(embed_size=self.embed_size).to(device)
        self.te_gnn = TemporalEncoder(embed_size=self.embed_size).to(device)
        self.te_head = TemporalEncoder(embed_size=self.obs_size).to(device)


    def train(self):
        set_random_seed(1234)

        graphs, obs_setting = load_dataset(self.data_path_graph, self.data_path_obs)

        T = 0
        losses = []
        epoch_losses = []

        self.model_gnn.train()
        self.model_head.train()
        optimizer = torch.optim.Adam(list(self.model_gnn.parameters()) + list(self.model_head.parameters()), lr=self.lr)
        optimizer.zero_grad()

        for epoch in range(self.epochs):
            indexes = np.random.permutation(len(graphs))
            pbar = tqdm(indexes, desc=f"Epoch {int(epoch)}/{self.epochs}", )

            for index in pbar:
                self.env.init_new_problem(index=index, setting_dict=obs_setting)

                points, edge_index, _, _, _, path_time, feasible, _, _ = graphs[index]

                if not feasible:
                    continue

                goal_state = self.env.goal_state
                goal_index = -1

                path_time = load_path_time(path_time, device)


                data = Data(goal=torch.FloatTensor(goal_state),
                            v=torch.FloatTensor(points))


                data.edge_index = torch.LongTensor(edge_index.T)

                current_loop = np.random.randint(1, self.loop)

                # create labels
                labels = torch.zeros(len(data.v), 1)
                labels[goal_index, 0] = 1

                ## get temporal encoding
                x = torch.arange(len(self.env.obs_points))
                pe = self.te_gnn(x).to(device)

                #### Stage1: Global Encoding ####
                edge_feat, node_feat = self.model_gnn(**data.to(device).to_dict(),
                                                 labels=labels.to(device),
                                                 pos_enc=pe,
                                                 obstacles=torch.FloatTensor(self.env.obs_points).to(device),
                                                 loop=current_loop)

                #### Stage2: Local Planning ####
                policy_loss = 0
                for step in range(path_time.shape[0] - 1):
                    source = path_time[step, 0]
                    end = path_time[step + 1, 0]

                    nonzero_indices = torch.where(edge_feat[source, :, :] != 0)[0].unique()
                    edges = edge_feat[source, nonzero_indices, :]

                    # edge:[N, embed_size],  obs:[N, obs_size]
                    time_window = list(range(-self.half_win_len, self.half_win_len+1))
                    offsets = torch.LongTensor(time_window).to(device)
                    time_window = torch.clip(path_time[step, 1] + offsets, 0, self.env.obs_points.shape[0] - 1)
                    obs = torch.FloatTensor(self.env.obs_points)[time_window].flatten()[None, :].repeat(edges.shape[0],
                                                                                                   1).to(device)
                    tmp_enc_tw = self.te_head(time_window).flatten()[None, :].repeat(edges.shape[0], 1).to(device)

                    policy = self.model_head(edges, obs, tmp_enc_tw)  # [N, 32*4]
                    policy_loss += -policy.log_softmax(dim=0)[torch.where(nonzero_indices == end)][
                        0]  # a variant of the cross entropy

                policy_loss.backward()
                losses.append((policy_loss))
                epoch_losses.append(policy_loss)

                if T % 10 == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss = [sum([loss[i] for loss in losses]) / len(losses) for i in range(1)]

                    pbar.set_description(f"Epoch {int(epoch)}/{self.epochs}, %.2f" % total_loss[0])

                    losses = []

                    torch.save(self.model_gnn.state_dict(), self.model_gnn_path)
                    torch.save(self.model_head.state_dict(), self.model_head_path)

                T += 1

            epoch_losses = []
            # save model
            if (epoch + 1) % 20 == 0:
                torch.save(self.model_gnn.state_dict(), self.model_gnn_path[:-3] + f'_epoch_{epoch}.pt')
                torch.save(self.model_head.state_dict(), self.model_head_path[:-3] + f'_epoch_{epoch}.pt')

        torch.save(self.model_gnn.state_dict(), self.model_gnn_path)
        torch.save(self.model_head.state_dict(), self.model_head_path)



if __name__ == '__main__':
    cfg = load_config(args.yaml_file)
    trainer = Trainer(cfg)
    trainer.train()
