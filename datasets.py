import warnings
import anndata
import numpy as np
import pandas as pd
import torch
from numpy.distutils.conv_template import header

from sympy.utilities.exceptions import ignore_warnings
from torch.nn.utils import skip_init
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import csv

warnings.filterwarnings("ignore")


class Her2st(Dataset):
    def __init__(self, args):
        self.mode = args.mode
        self.spot_to_slide = []
        self.slide_global_ebd = {}
        if self.mode == "train":
            slide_name_list = list(np.genfromtxt(args.data_path + 'processed_data/' + args.slidename_list, dtype='str'))
            for slide_out in args.slide_out.split(','):
                slide_name_list.remove(slide_out)
                print(f"{slide_out} is removed for testing")
            selected_genes = list(np.genfromtxt(args.data_path + 'processed_data/' + args.gene_list, dtype='str'))
            args.input_gene_size = len(selected_genes)
            print(f'len selected genes : {len(selected_genes)}')

            first = True
            all_pos = []
            all_gene_mtx = None
            all_local_ebd = None
            all_neighbor_ebd = None
            all_global_ebd = None
            all_neighbor_pos = []
            all_global_pos = []
            for i in range(len(slide_name_list)):
                slide = slide_name_list[i]
                gene_mtx_ = anndata.read_h5ad(args.data_path + 'st/' + slide + '.h5ad')
                gene_mtx = pd.DataFrame(gene_mtx_[:, selected_genes].X.toarray(),
                                        columns=selected_genes,
                                        index=[slide + '_' + str(j) for j in range(gene_mtx_.shape[0])])
                spot_num = len(gene_mtx)

                pos_df = pd.read_csv(args.data_path + 'processed_data/pos/' + slide + '.csv', header=None)
                x = pos_df.iloc[1].values.tolist()
                y = pos_df.iloc[2].values.tolist()
                all_pos.append(list(zip(x, y)))

                if first:
                    all_gene_mtx = gene_mtx
                    local_ebd = torch.load(args.data_path + 'processed_data/local_ebd/' + slide + '.pt')
                    neighbor_ebd = torch.load(args.data_path + 'processed_data/neighbor_ebd/' + slide + '.pt')
                    global_ebd = torch.load(args.data_path + 'processed_data/global_ebd/' + slide + '.pt')
                    all_local_ebd = local_ebd
                    all_neighbor_ebd = neighbor_ebd
                    all_global_ebd = global_ebd
                    self.slide_global_ebd[slide] = local_ebd
                    for _ in range(spot_num):
                        neighbor_spots = find_nearst_k_spots(args.data_path + 'processed_data/pos/' + slide + '.csv',
                                                             x[_], y[_], k=9)
                        global_spots = find_nearst_k_spots(args.data_path + 'processed_data/pos/' + slide + '.csv',
                                                           x[_], y[_], k=49)
                        all_neighbor_pos.append(neighbor_spots)
                        all_global_pos.append(global_spots)
                    first = False
                    continue
                local_ebd = torch.load(args.data_path + 'processed_data/local_ebd/' + slide + '.pt')
                neighbor_ebd = torch.load(args.data_path + 'processed_data/neighbor_ebd/' + slide + '.pt')
                global_ebd = torch.load(args.data_path + 'processed_data/global_ebd/' + slide + '.pt')
                all_local_ebd = torch.cat([all_local_ebd, local_ebd], axis=0)
                all_neighbor_ebd = torch.cat([all_neighbor_ebd, neighbor_ebd], axis=0)
                all_gene_mtx = np.concatenate([all_gene_mtx, gene_mtx], axis=0)
                all_global_ebd = torch.cat([all_global_ebd, global_ebd], axis=0)
                self.slide_global_ebd[slide] = local_ebd
                for _ in range(spot_num):
                    neighbor_spots = find_nearst_k_spots(args.data_path + 'processed_data/pos/' + slide + '.csv', x[_],
                                                         y[_], k=9)
                    global_spots = find_nearst_k_spots(args.data_path + 'processed_data/pos/' + slide + '.csv', x[_],
                                                       y[_], k=49)
                    all_neighbor_pos.append(neighbor_spots)
                    all_global_pos.append(global_spots)
                print(
                    f"{slide} loaded, gene mtx shape: {all_gene_mtx.shape}, img ebd shape:{all_local_ebd.shape}, neighbor ebd shape:{all_neighbor_ebd.shape}, "
                    )

            args.cond_size = all_local_ebd.shape[1]
            all_pos = [coord for slide_pos in all_pos for coord in slide_pos]
            all_gene_mtx_df = pd.DataFrame(all_gene_mtx, columns=selected_genes,
                                           index=list(range(all_gene_mtx.shape[0])))
            # remove the spot with all NAN/zero in gene mtx
            all_gene_mtx_all_nan_spot_index = all_gene_mtx_df.index[all_gene_mtx_df.isnull().all(axis=1)]
            all_gene_mtx_all_zero_spot_index = all_gene_mtx_df.index[all_gene_mtx_df.sum(axis=1) == 0]
            print(f"All NAN spot index: {all_gene_mtx_all_nan_spot_index}")
            print(f"All zero spot index: {all_gene_mtx_all_zero_spot_index}")
            spot_idx_to_remove = list(set(all_gene_mtx_all_nan_spot_index) | set(all_gene_mtx_all_zero_spot_index))
            spot_idx_to_keep = list(set(all_gene_mtx_df.index) - set(spot_idx_to_remove))
            all_pos = [item for i, item in enumerate(all_pos) if i not in spot_idx_to_remove]
            all_neighbor_pos = [item for i, item in enumerate(all_neighbor_pos) if i not in spot_idx_to_remove]
            all_global_pos = [item for i, item in enumerate(all_global_pos) if i not in spot_idx_to_remove]
            all_gene_mtx = all_gene_mtx[spot_idx_to_keep, :]
            all_gene_mtx_selected_genes = np.log2(all_gene_mtx + 1).copy()
            all_local_ebd = all_local_ebd[spot_idx_to_keep, :]
            all_neighbor_ebd = all_neighbor_ebd[spot_idx_to_keep, :]
            all_global_ebd = all_global_ebd[spot_idx_to_keep, :]
        else:
            slide_name_list = args.slide_out
            selected_genes = list(np.genfromtxt(args.data_path + 'processed_data/' + args.gene_list, dtype='str'))
            args.input_gene_size = len(selected_genes)
            print(f'len selected genes : {len(selected_genes)}')
            first = True
            all_pos = []
            all_gene_mtx = None
            all_local_ebd = None
            all_neighbor_ebd = None
            all_global_ebd = None
            all_neighbor_pos = []
            all_global_pos = []
            slide = args.slide_out
            gene_mtx_ = anndata.read_h5ad(args.data_path + 'st/' + slide + '.h5ad')
            gene_mtx = pd.DataFrame(gene_mtx_[:, selected_genes].X.toarray(),
                                    columns=selected_genes,
                                    index=[slide + '_' + str(j) for j in range(gene_mtx_.shape[0])])
            spot_num = len(gene_mtx)

            pos_df = pd.read_csv(args.data_path + 'processed_data/pos/' + slide + '.csv', header=None)
            x = pos_df.iloc[1].values.tolist()
            y = pos_df.iloc[2].values.tolist()
            all_pos.append(list(zip(x, y)))

            all_gene_mtx = gene_mtx
            local_ebd = torch.load(args.data_path + 'processed_data/local_ebd/' + slide + '.pt')
            neighbor_ebd = torch.load(args.data_path + 'processed_data/neighbor_ebd/' + slide + '.pt')
            global_ebd  = torch.load(args.data_path + 'processed_data/global_ebd/' + slide + '.pt')
            all_local_ebd = local_ebd
            all_neighbor_ebd = neighbor_ebd
            all_global_ebd = global_ebd
            for _ in range(spot_num):
                neighbor_spots = find_nearst_k_spots(args.data_path + 'processed_data/pos/' + slide + '.csv', x[_],y[_],k=9)
                global_spots = find_nearst_k_spots(args.data_path + 'processed_data/pos/' + slide + '.csv',x[_],y[_],k=49)
                all_neighbor_pos.append(neighbor_spots)
                all_global_pos.append(global_spots)

            print(
                f"{slide} loaded, gene mtx shape: {all_gene_mtx.shape}, img ebd shape:{all_local_ebd.shape}, neighbor ebd shape:{all_neighbor_ebd.shape}, "
                )

            args.cond_size = all_local_ebd.shape[1]
            all_pos = [coord for slide_pos in all_pos for coord in slide_pos]
            all_gene_mtx_df = pd.DataFrame(all_gene_mtx, columns=selected_genes,
                                           index=list(range(all_gene_mtx.shape[0])))
            all_gene_mtx_selected_genes = np.log2(all_gene_mtx + 1).copy().to_numpy()
        self.gene_mtx = all_gene_mtx_selected_genes
        self.local_ebd = all_local_ebd
        self.neighbor_ebd = all_neighbor_ebd
        self.neighbor_pos = torch.tensor(all_neighbor_pos, dtype=torch.float32)
        self.all_pos = all_pos
        self.global_ebd = all_global_ebd
        self.global_pos = torch.tensor(all_global_pos, dtype=torch.float32)
        self.args = args



    def __len__(self):
        return len(self.gene_mtx)


    def __getitem__(self, idx):
        gene = self.gene_mtx[idx]
        local_ebd = self.local_ebd[idx]
        neighbor_ebd = self.neighbor_ebd[idx]
        global_ebd = self.global_ebd[idx]
        # slide_id = self.spot_to_slide[idx]
        pos = self.all_pos[idx]
        neighbor_pos = self.neighbor_pos[idx]
        global_pos = self.global_pos[idx]
        return gene, local_ebd, neighbor_ebd, global_ebd, pos, neighbor_pos, global_pos


    def get_args(self):
        return self.args


def find_nearst_k_spots(csv_path, x, y, k=9):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    x_row = list(map(float, rows[1]))
    y_row = list(map(float, rows[2]))
    points = np.array(list(zip(x_row, y_row)))
    target = np.array([x, y])
    distances = np.linalg.norm(points - target, axis=1)
    nearest_coords = points[np.argsort(distances)[:k]]
    return nearest_coords.tolist()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # data related arguments
    parser.add_argument("--expr_name", type=str, default="her2st")
    parser.add_argument("--data_path", type=str, default="./hest1k_datasets/her2st/", help="Dataset path")
    parser.add_argument("--results_dir", type=str, default="./her2st_results/runs/", help="Path to hold runs")
    parser.add_argument("--slide_out", type=str, default="SPA152",
                        help="Test slide ID. Multiple slides separated by comma.")
    parser.add_argument("--slidename_list", type=str, default="all_slide_lst.txt",
                        help="A txt file listing file names for all training and testing slides in the dataset")
    parser.add_argument("--gene_list", type=str, default="selected_gene_list.txt", help="Selected gene list")
    parser.add_argument("--mode", type=str, default="train", help="Running mode (train/test)")
    parser.add_argument("--num_aug_ratio", type=int, default=7)
    # model related arguments
    parser.add_argument("--model", type=str, default="Stem")
    parser.add_argument("--DiT_num_blocks", type=int, default=12, help="DiT depth")
    parser.add_argument("--hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=6, help="DiT heads")
    parser.add_argument("--device", type=str, default='cuda:0', help="Gpu")
    # training related arguments
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--ckpt_every", type=int, default=50, help="Number of epoch to save checkpoints.")

    input_args = parser.parse_args()

    d = Her2st(input_args)
