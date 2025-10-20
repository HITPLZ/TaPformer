import os
import numpy as np

import sys
import networkx as nx
import yaml
import argparse
import configparser

# from karateclub import GraRep, HOPE
# from node2vec import Node2Vec

# from umap import UMAP

sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

torch.set_num_threads(3)

from src.models.Patch.Patch_wBias import TaPformer


from src.base.engine import BaseEngine
# from src.base.engine_auto import BaseEngine


from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info, load_patch_indices_with_padding
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()


    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--learning_rate', type=float, default = 1e-3)
    parser.add_argument('--weight_decay', type=float, default = 5e-5)
    parser.add_argument('--clip_grad_value', type=float, default=5)

    parser.add_argument('--input_len', type = int, default = 12)
    parser.add_argument('--output_len', type = int, default = 12)
    parser.add_argument('--his_len', type = int, default = 96 * 14)
    parser.add_argument('--train_ratio', type = float, default = 0.6)
    parser.add_argument('--val_ratio', type = float, default = 0.2)
    parser.add_argument('--test_ratio', type = float, default = 0.2)

    parser.add_argument('--layers', type=int, default = 4)
    parser.add_argument('--tem_patchsize', type = int, default = 12)
    parser.add_argument('--tem_patchnum', type = int, default = 1)
    parser.add_argument('--node_num', type = int, default = 64)
    parser.add_argument('--tod', type=int, default = 96)
    parser.add_argument('--dow', type=int, default = 7)
    parser.add_argument('--input_dims', type=int, default = 32)
    parser.add_argument('--node_dims', type=int, default = 64)
    parser.add_argument('--tod_dims', type=int, default = 32)
    parser.add_argument('--dow_dims', type=int, default = 32)
    parser.add_argument('--patch_num', type=int, default = 32)
    parser.add_argument('--patch_inner', type=int, default = 32)


    parser.add_argument('--patch_file', default=None)
    parser.add_argument('--sub_adj_file', default=None)
    #only for PatchSTG
    # parser.add_argument('--ori_parts_idx', default=None)
    # parser.add_argument('--reo_all_idx', default=None)
    # parser.add_argument('--reo_parts_idx', default=None)




    args = parser.parse_args()
    with open(f"./config/Patch.yaml", "r") as f:
        cfg = yaml.safe_load(f)[args.dataset]
    vars(args).update(cfg)
    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    dataloader, scaler = load_dataset(data_path, args, logger)

    ori_parts_idx, reo_parts_idx, reo_all_idx, mask_tensor, patch_adj_matrices = \
        load_patch_indices_with_padding(args.patch_file, args.sub_adj_file)

    #only for patchSTG
    # ori_parts_idx = np.load(args.ori_parts_idx)
    # reo_parts_idx = np.load(args.reo_parts_idx)
    # reo_all_idx = np.load(args.reo_all_idx)


    # node_num = node_num,
    # input_dim = args.input_dim,
    # output_dim = args.output_dim,
    model = TaPformer(
                      ori_parts_idx = ori_parts_idx,
                      reo_parts_idx = reo_parts_idx,
                      reo_all_idx = reo_all_idx,
                      mask_tensor = mask_tensor,
                      patch_adj_matrices = patch_adj_matrices,
                      node_num=node_num,
                      input_dim=args.input_dim,
                      output_dim=args.output_dim,
                      model_args=vars(args),
                      supports=supports)

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps = [1, 35, 40]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.5)

    engine = BaseEngine(device=device,
                        model=model,
                        dataloader=dataloader,
                        scaler=scaler,
                        sampler=None,
                        loss_fn=loss_fn,
                        lrate=args.learning_rate,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        clip_grad_value=args.clip_grad_value,
                        max_epochs=args.max_epoch,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        seed=1,
                        )
    args.mode = 'test'
    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
