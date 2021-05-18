import numpy as np
import os
from scipy import sparse
import pandas as pd
import random
from model import VAE
from multiomicDataset import collate_wrapper, MultiomicDataset, shuffle_dataloaders
from load_data import load_merged_remapped
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
torch.cuda.empty_cache()

# ====================================
# ==================================== read data
# ====================================

# Check if results are stable to quality filtering
# F: no filtering
# M: minial filtering
# P: partial filtering
# T: filter to annotated cells
is_filter = "F"
output_dir = "../output/20210210_" + is_filter + "/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

if is_filter == "T":
    filter_cells = True
elif is_filter == "P":
    filter_cells = False
    cell_file = {
        "rna": "rna_cells_1000.txt",
        "atac": "atac_cells_200.txt"
    }
elif is_filter == "M":
    filter_cells = False
    cell_file = {
        "rna": "rna_cells_200.txt",
        "atac": "atac_cells_100.txt"
    }
elif is_filter == "F":
    filter_cells = False


multi_dt = load_merged_remapped(10000000,
                                gene_filter_lower=None, gene_filter_upper=None,
                                peak_filter_lower=None, peak_filter_upper=None,
                                filter_cells=filter_cells, filter_features=True)

# ====================================
# ==================================== quality filtering
# ====================================
if is_filter in ["P", "M"]:
    barcode_dir = "../../multiomics_clustering/rcode/20210204_method_comparison/quality_cells"
    for omic in ["rna", "atac"]:
        cells = pd.read_csv(
            os.path.join(barcode_dir, cell_file[omic]),
            header=None, sep="\t")[0].values.astype('str')
        bool_cells = (np.isin(multi_dt[omic]['barcode'], cells)) | \
                   (multi_dt[omic]['dataset'] == 0)
        multi_dt[omic]['counts'] = multi_dt[omic]['counts'][bool_cells, ]
        multi_dt[omic]['barcode'] = multi_dt[omic]['barcode'][bool_cells]
        multi_dt[omic]['dataset'] = multi_dt[omic]['dataset'][bool_cells]

multi_dt = MultiomicDataset(dt=multi_dt)
barcode = multi_dt.get_barcode()
# ------------------------------ create train test split
np.random.seed(0)
n_samples = len(multi_dt)
permuted_idx = np.random.permutation(n_samples)
train_idx = permuted_idx[:int(n_samples*1)]
# test_idx = permuted_idx[int(n_samples*1):] # Here we do not use test data
train_barcode = barcode[train_idx]
# test_barcode = barcode[test_idx]
np.savetxt(os.path.join(output_dir, "train_barcode.txt"), train_barcode, fmt="%s")
# np.savetxt("../output/test_barcode.txt", test_barcode, fmt="%s")
np.savetxt(os.path.join(output_dir, "barcode.txt"), barcode, fmt="%s")

# ====================================
# ==================================== Running
# ====================================

num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_latent_list = [30]
current_run = 0
kl_penalty = 0
total_run = len(n_latent_list)
while current_run < total_run:
    n_latent = n_latent_list[current_run]
    DIVERGE = False
    print("======================= Current Run", current_run)
    learning_rate = 0.01
    annealing_epochs = 30
    alpha = 2
    hidden_dims = [128]
    file_sub = "w_single_omic_" + str(kl_penalty) + '_' + str(annealing_epochs) \
               + '_' + str(alpha) + '_' + str(n_latent)
    model = VAE(in_channels=multi_dt.get_feature_shape(),
                hidden_dims=hidden_dims,
                n_dataset=multi_dt.n_dataset,
                latent_dim=n_latent,
                intercept_adj=True,
                slope_adj=True,
                log=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"loss": []}

    # Start training
    for epoch in range(1, num_epochs + 1):
        print('epoch' + str(epoch))
        if epoch < annealing_epochs:
            annealing_factor = float(epoch) / float(annealing_epochs)
        else:
            annealing_factor = 1.0

        latent_loss_all = []
        recon_loss_all = []
        this_loss = []
        train_omic = [[True, True], [True, False], [False, True]]
        train_loaders = [
            DataLoader(
                dataset=multi_dt,
                batch_size=128,
                collate_fn=lambda x: collate_wrapper(x, [True, True]),
                sampler=SubsetRandomSampler(
                    np.intersect1d(multi_dt.get_comb_idx([True, True]), train_idx)
                )),
            DataLoader(
                dataset=multi_dt,
                batch_size=128,
                collate_fn=lambda x: collate_wrapper(x, [True, False]),
                sampler=SubsetRandomSampler(
                    np.intersect1d(multi_dt.get_comb_idx([True, False]), train_idx)
                )),
            DataLoader(
                dataset=multi_dt,
                batch_size=128,
                collate_fn=lambda x: collate_wrapper(x, [False, True]),
                sampler=SubsetRandomSampler(
                    np.intersect1d(multi_dt.get_comb_idx([False, True]), train_idx)
                ))
        ]
        loaders_sum = sum([len(x) for x in train_loaders])
        objective_weights = {
            (True, True): loaders_sum / len(train_loaders[0]),
            (True, False): loaders_sum / len(train_loaders[1]),
            (False, True): loaders_sum / len(train_loaders[2])
        }

        shuffled_loaders = shuffle_dataloaders(train_loaders, train_omic)

        for x in shuffled_loaders:
            x, omic_combn = x
            # Forward pass
            x = [[x_i.to(device) if x_i is not None else None for x_i in y] for y in x]
            latent_loss, recon_loss = model(x, elbo_combn=[omic_combn], kl_penalty=kl_penalty)
            # Backprop and optimize
            loss = (annealing_factor * latent_loss + recon_loss) * objective_weights[(omic_combn[0], omic_combn[1])]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            latent_loss_all += [latent_loss.item()]
            recon_loss_all += [recon_loss.item()]

        this_loss += [np.array(latent_loss_all) + np.array(recon_loss_all)]

        history['loss'] += [np.concatenate(this_loss)]
        if np.isnan(history['loss'][-1].mean()):
            DIVERGE = True
            break
        # print([x.mean() / 128 for x in history['loss']])
    if DIVERGE:
        continue

    with torch.no_grad():
        # === save latent variables
        for omic_combn in [[True, False], [False, True], [True, True]]:
            for dt_str, sample_idx in zip(['train'], [train_idx]):
            # for dt_str, sample_idx in zip(['train', 'test'], [train_idx, test_idx]):
                sample_idx = np.intersect1d(multi_dt.get_comb_idx(omic_combn), sample_idx)
                dl = DataLoader(
                    dataset=Subset(multi_dt, sample_idx),
                    batch_size=128,
                    collate_fn=lambda x: collate_wrapper(x, omic_combn),
                    shuffle=False
                )

                latent_test = []
                for i, x in enumerate(dl):
                    x = [[x_i.to(device) if x_i is not None else None for x_i in y] for y in x]
                    latent_test += [model.get_latent(x, elbo_bool=omic_combn)]
                res = np.concatenate(latent_test)
                np.savetxt(output_dir + file_sub +
                           "_".join([str(x) for x in omic_combn]) +
                           "_" + dt_str + "_latent.csv",
                           res, delimiter=",")

                np.savetxt(output_dir + file_sub +
                           "_".join([str(x) for x in omic_combn]) +
                           "_" + dt_str + "_barcode.csv",
                           barcode[sample_idx], fmt="%s")
        plt.close()
        # === save feature
        for i, x in enumerate(model.get_beta()):
            np.savetxt(output_dir + file_sub + "beta" + str(i) + ".csv",
                       x, delimiter=",")
        res.sum(axis = 0)

        current_run += 1
