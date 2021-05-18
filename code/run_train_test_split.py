import numpy as np
import os
from scipy import sparse
from lda_prod_joint_dt_specific import VAE
from load_data import load_merged_remapped, load_snare_seq, load_snare_seq_genes, load_snare_seq_remapped
from multiomicDataset import collate_wrapper, MultiomicDataset, shuffle_dataloaders
import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
torch.cuda.empty_cache()

# ====================================
# ==================================== read data
# ====================================

output_dir = "../output/train_test_split/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

counts, features, barcodes = load_snare_seq("Ad")
# counts, features, barcodes = load_snare_seq_remapped("Ad")
# counts, features, barcodes = load_snare_seq_genes("Ad")
n_samples = len(barcodes)
np.random.seed(0)
TRAINING_PROP = 0.2
permuted_idx = np.random.permutation(n_samples)
train_idx = permuted_idx[:int(n_samples*TRAINING_PROP)]
test_idx = permuted_idx[int(n_samples*TRAINING_PROP):]

train_barcodes = barcodes[train_idx]
test_barcodes = barcodes[test_idx]
np.savetxt(os.path.join(output_dir, "train_barcode.txt"), train_barcodes, fmt="%s")
np.savetxt(os.path.join(output_dir, "test_barcode.txt"), test_barcodes, fmt="%s")
np.savetxt(os.path.join(output_dir, "barcode.txt"), barcodes, fmt="%s")
np.savetxt(os.path.join(output_dir, "genes.txt"), features['rna'], fmt="%s")
np.savetxt(os.path.join(output_dir, "peaks.txt"), features['atac'], fmt="%s")

multi_dt = {
    'rna': {
        'feature': features['rna'],
        'counts': counts['rna'],
        'barcode': np.array([b if i in train_idx else b + "_rna" for i, b in enumerate(barcodes)]),
        'dataset': np.array([0 if i in train_idx else 1 for i, b in enumerate(barcodes)])
    },
    'atac': {
        'feature': features['atac'],
        'counts': counts['atac'],
        'barcode': np.array([b if i in train_idx else b + "_atac" for i, b in enumerate(barcodes)]),
        'dataset': np.array([0 if i in train_idx else 1 for i, b in enumerate(barcodes)])
    }
}

multi_dt = MultiomicDataset(dt=multi_dt)
barcode = multi_dt.get_barcode()

# ====================================
# ==================================== running
# ====================================

num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_latent_list = [10]
current_run = 0
kl_penalty = 0
total_run = len(n_latent_list)
while current_run < total_run:
    n_latent = n_latent_list[current_run]
    DIVERGE = False
    print("======================= Current Run", current_run)
    learning_rate = 0.005
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
                sampler=SubsetRandomSampler(np.asarray(multi_dt.get_comb_idx([True, True])[0])
                )),
            DataLoader(
                dataset=multi_dt,
                batch_size=128,
                collate_fn=lambda x: collate_wrapper(x, [True, False]),
                sampler=SubsetRandomSampler(np.array(multi_dt.get_comb_idx([True, False])[0])
                )),
            DataLoader(
                dataset=multi_dt,
                batch_size=128,
                collate_fn=lambda x: collate_wrapper(x, [False, True]),
                sampler=SubsetRandomSampler(np.array(multi_dt.get_comb_idx([False, True])[0])
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
            for dt_str in ['train']:
                sample_idx = multi_dt.get_comb_idx(omic_combn)[0]
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

        current_run += 1
