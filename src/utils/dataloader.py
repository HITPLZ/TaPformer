import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp
import json


class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, his_len, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)

        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon
        self.his_len = his_len
        self.his_offsets = np.arange(-(self.his_len - 1), 1, 1)
        self.his_mask = np.zeros((self.his_len, self.data.shape[1], self.data.shape[2]))

    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx

    def write_to_shared_array(self, x, y, his, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]
            if idx_ind[i] - self.his_len < 0:
                his[i] = self.his_mask
            else:
                his[i] = self.data[idx_ind[i] + self.his_offsets, :, :]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                his_shape = (len(idx_ind), self.his_len, self.data.shape[1], self.data.shape[-1])
                his_shared = mp.RawArray('f', int(np.prod(his_shape)))
                his = np.frombuffer(his_shared, dtype='f').reshape(his_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array,
                                              args=(x, y, his, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y, his)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(data_path, args, logger):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    logger.info('Data shape: ' + str(ptr['data'].shape))

    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx,
                                                 args.seq_len, args.horizon, args.his_len, args.bs, logger)

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
        'CA': [base_dir + 'ca', base_dir + 'ca/ca_rn_adj.npy', 8600],
        'GLA': [base_dir + 'gla', base_dir + 'gla/gla_rn_adj.npy', 3834],
        'GBA': [base_dir + 'gba', base_dir + 'gba/gba_rn_adj.npy', 2352],
        'SD': [base_dir + 'sd', base_dir + 'sd/sd_rn_adj.npy', 716],
    }
    assert dataset in d.keys()
    return d[dataset]

def load_patch_indices_with_padding(json_path,
                                    patch_adj_matrices_path,
                                    ):
    """
    1) 从 json_path 读取 {patch_id: [node_idx,...]}，
    2) 从 patch_adj_matrices_path 读取 {patch_id: [[...],[...],...]} 的子邻接矩阵，
    3) 对所有 patch 做 padding，使得节点数都 = N_max，
    4) 返回：
        ori_parts_idx:   List[List[int]]        原始每个 patch 的全局节点索引
        reo_parts_idx:   List[List[int]]        重排后 patch 内的新序号
        reo_all_idx:     List[int]              所有 patch 拼接后的全局节点索引（含 padding）
        mask_tensor:     Tensor[P, N_max]       padding mask（1=有效，0=pad）
        patch_adj_tensor:Tensor[P, N_max, N_max] 子邻接矩阵（pad 后）
    """
    # 1) 读 patch_nodes
    with open(json_path, 'r') as f:
        patch_dict = json.load(f)
    # 按 patch_id 排序（保证和 adj 矩阵顺序对齐）
    keys = sorted(patch_dict, key=lambda x: int(x))
    ori_parts_idx = [patch_dict[k] for k in keys]
    P = len(ori_parts_idx)

    # 找到最大 patch 长度
    N_max = max(len(nodes) for nodes in ori_parts_idx)

    # 2) padding + mask
    padded_parts = []
    masks = []
    for nodes in ori_parts_idx:
        L = len(nodes)
        padded = nodes + [0] * (N_max - L)
        mask = [1] * L + [0] * (N_max - L)
        padded_parts.append(padded)
        masks.append(mask)
    mask_tensor = torch.tensor(masks, dtype=torch.float32)  # [P, N_max]

    # 3) 扁平化索引 & 重排
    reo_all_idx = [idx for part in padded_parts for idx in part]
    idx_map = {orig: new for new, orig in enumerate(reo_all_idx)}
    reo_parts_idx = [[idx_map[n] for n in part] for part in padded_parts]

    # 4) 读子邻接矩阵，并 pad 到 [N_max, N_max]
    with open(patch_adj_matrices_path, 'r') as f:
        patch_adj_raw = json.load(f)
    # 保证顺序与 keys 一致
    patch_adj_list = [np.array(patch_adj_raw[k], dtype=np.float32) for k in keys]
    patch_adj_padded = []
    for sub_adj in patch_adj_list:
        n = sub_adj.shape[0]
        if n < N_max:
            # pad 右下角补 0
            pad = np.zeros((N_max, N_max), dtype=np.float32)
            pad[:n, :n] = sub_adj
            patch_adj_padded.append(pad)
        else:
            patch_adj_padded.append(sub_adj[:N_max, :N_max])
    patch_adj_tensor = torch.tensor(patch_adj_padded)  # [P, N_max, N_max]

    return ori_parts_idx, reo_parts_idx, reo_all_idx, mask_tensor, patch_adj_tensor

def augmentAlign(dist_matrix, auglen):
    # find the most similar points in other leaf nodes
    sorted_idx = np.argsort(dist_matrix.reshape(-1)*-1)
    sorted_idx = sorted_idx % dist_matrix.shape[-1]
    augidx = []
    for idx in sorted_idx:
        if idx not in augidx:
            augidx.append(idx)
        if len(augidx) == auglen:
            break
    return np.array(augidx, dtype=int)

def reorderData(parts_idx, mxlen, adj, sps):
    # parts_idx: segmented indices by kdtree
    # adj: pad similar points through the cos_sim adj
    # sps: spatial patch (small leaf nodes) size for padding
    ori_parts_idx = np.array([], dtype=int)
    reo_parts_idx = np.array([], dtype=int)
    reo_all_idx = np.array([], dtype=int)
    for i, part_idx in enumerate(parts_idx):
        part_dist = adj[part_idx, :].copy()
        part_dist[:, part_idx] = 0
        if sps-part_idx.shape[0] > 0:
            local_part_idx = augmentAlign(part_dist, sps-part_idx.shape[0])
            auged_part_idx = np.concatenate([part_idx, local_part_idx], 0)
        else:
            auged_part_idx = part_idx

        reo_parts_idx = np.concatenate([reo_parts_idx, np.arange(part_idx.shape[0])+sps*i])
        ori_parts_idx = np.concatenate([ori_parts_idx, part_idx])
        reo_all_idx = np.concatenate([reo_all_idx, auged_part_idx])

    return ori_parts_idx, reo_parts_idx, reo_all_idx

def kdTree(locations, times, axis):
    # locations: [2,N] contains lng and lat
    # times: depth of kdtree
    # axis: select lng or lat as hyperplane to split points
    sorted_idx = np.argsort(locations[axis])
    part1, part2 = np.sort(sorted_idx[:locations.shape[1]//2]), np.sort(sorted_idx[locations.shape[1]//2:])
    parts = []
    if times == 1:
        return [part1, part2], max(part1.shape[0], part2.shape[0])
    else:
        left_parts, lmxlen = kdTree(locations[:,part1], times-1, axis^1)
        right_parts, rmxlen = kdTree(locations[:,part2], times-1, axis^1)
        for part in left_parts:
            parts.append(part1[part])
        for part in right_parts:
            parts.append(part2[part])
    return parts, max(lmxlen, rmxlen)