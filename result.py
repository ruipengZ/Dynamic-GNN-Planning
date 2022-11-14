import pickle
import numpy as np

def show_result(model_name, gt_file=None, result_dict=None, result_file=None, index_list=None):
    graphs = []
    for file_idx in range(len(gt_file)):
        with open(gt_file[file_idx], 'rb') as f:
            load_graph = pickle.load(f)
            graphs.extend(load_graph)


    if index_list is not None:
        graphs = [graphs[i] for i in index_list]

    gt_time = []
    gt_collision_checking = []
    for item in graphs:
        gt_time.append(item[-2])
        gt_collision_checking.append(item[-1])

    gt_time = np.array(gt_time)

    if result_file:
        with np.load(result_file, allow_pickle=True) as f:
            success = f['success']
            path_time = f['path_time']
            path = f['path']
            check_collision = f['check_collision']
            inference_time = f['inference_time']
    else:
        success = np.array(result_dict['success'])
        path_time = np.array(result_dict['path_time'])
        check_collision = np.array(result_dict['check_collision'])
        inference_time = np.array(result_dict['inference_time'])


    success_rate = np.sum(success)/success.shape[0]
    average_gt_time = np.mean(gt_time)
    average_gt_collision_checking = np.mean(gt_collision_checking)
    std_gt_collision_checking = np.std(gt_collision_checking)
    median_gt_collision_checking = np.median(gt_collision_checking)

    average_gnn_time = np.mean(path_time[np.where(success!=False)])
    percentage_time = path_time[np.where(success != False)] / gt_time[np.where(success != False)]
    average_percentage_time = np.mean(percentage_time)

    average_collision_checking = np.mean(check_collision[np.where(success!=False)])
    std_collision_checking = np.std(check_collision[np.where(success != False)])
    median_collision_checking = np.median(check_collision[np.where(success != False)])

    percentage_collision_checking = check_collision[np.where(success != False)] / np.array(gt_collision_checking)[
        np.where(success != False)[0]]
    average_collision_checking_pc = np.mean(percentage_collision_checking)
    std_collision_checking_pc = np.std(percentage_collision_checking)
    median_collision_checking_pc = np.median(percentage_collision_checking)

    average_inference_time = np.mean(inference_time)
    print('model_name:', model_name)
    print('average_gt_time', average_gt_time)
    print('average_gt_collision_checking', average_gt_collision_checking)
    print('std_gt_collision_checking', std_gt_collision_checking)
    print('median_gt_collision_checking', median_gt_collision_checking)

    print('success_rate', success_rate)
    print('average_gnn_time', average_gnn_time)
    print('average_percentage_time', average_percentage_time)
    print('average_collision_checking', average_collision_checking)
    print('std_collision_checking', std_collision_checking)
    print('median_collision_checking', median_collision_checking)

    print('average_collision_checking_percentage', average_collision_checking_pc)
    print('std_collision_checking_percentage', std_collision_checking_pc)
    print('median_collision_checking_percentage', median_collision_checking_pc)

    print('average_inference_time', average_inference_time)