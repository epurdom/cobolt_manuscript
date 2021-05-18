import numpy as np
from model import VAE
from load_data import load_snare_seq
import torch
import os
from torch.utils.data import DataLoader, Subset

dataset = "Ad"

class CoboltDataset(torch.utils.data.Dataset):
    def __init__(self, X,
                 features=None,
                 barcodes=None):
        self.X = X
        self.features = features
        self.barcodes = barcodes

    def __len__(self):
        """Number of samples in the data"""
        return self.X[0].shape[0]

    def __getitem__(self, index):
        """Generates one sample of data"""
        dat = [torch.from_numpy(dt[index].toarray()) for dt in self.X]
        return dat

    def get_feature_shape(self):
        return [y.shape[1] for y in self.X]

def collate_wrapper(batch):
    return [torch.cat(i).float() for i in zip(*batch)]


# ====================================
# ==================================== read data
# ====================================

counts, features, barcodes = load_snare_seq(dataset)
n_samples = len(barcodes)
np.random.seed(0)
permuted_idx = np.random.permutation(n_samples)
train_idx = permuted_idx[:int(n_samples*1)]
test_idx = permuted_idx[int(n_samples*1):]

train_barcodes = barcodes[train_idx]
test_barcodes = barcodes[test_idx]

output_dir = os.path.join("../output/", "snare_only")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

np.savetxt(os.path.join(output_dir, "train_barcodes.txt"), train_barcodes, fmt="%s")
np.savetxt(os.path.join(output_dir, "test_barcodes.txt"), test_barcodes, fmt="%s")

np.savetxt(os.path.join(output_dir, "genes.txt"), features['rna'], fmt="%s")
np.savetxt(os.path.join(output_dir, "peaks.txt"), features['atac'], fmt="%s")


dt = CoboltDataset(X=list(counts.values()),
                            features=features,
                            barcodes=barcodes)
train_dt = Subset(dt, train_idx)
train_loader = DataLoader(dataset=train_dt,
                          batch_size=128,
                          shuffle=False,
                          collate_fn=collate_wrapper)
test_dt = Subset(dt, test_idx)
test_loader = DataLoader(dataset=test_dt,
                          batch_size=128,
                          shuffle=False,
                          collate_fn=collate_wrapper)

# ====================================
# ==================================== Runnings
# ====================================

elbo_comb = [[True, False], [False, True], [True, True]]
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


n_latent = 10
total_run = 1
current_run = 0
while current_run < total_run:
    print("======================= Current Run", current_run)
    learning_rate = 0.005
    annealing_epochs = 30
    alpha = 2
    hidden_dims = [128]
    file_sub = dataset + str(learning_rate) + '_' + str(annealing_epochs) \
               + '_' +  str(alpha) + '_' + str(n_latent)
    model = VAE(in_channels=dt.get_feature_shape(), hidden_dims = hidden_dims, latent_dim=n_latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"loss": []}

    # Start training
    for epoch in range(1, num_epochs + 1):
        print('epoch' + str(epoch))
        latent_loss_all = []
        recon_loss_all = []
        for i, x in enumerate(train_loader):
            if epoch < annealing_epochs:
                # compute the KL annealing factor for the current mini-batch in the current epoch
                annealing_factor = (
                        float(i + (epoch - 1) * len(train_loader) + 1) /
                        float(annealing_epochs * len(train_loader)))
            else:
                # by default the KL annealing factor is unity
                annealing_factor = 1.0
            # Forward pass
            x = [x_i.to(device) for x_i in x]
            latent_loss, recon_loss = model(x, elbo_combn=elbo_comb, kl_penalty=0)
            # Backprop and optimize
            loss = annealing_factor * latent_loss + recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            latent_loss_all += [latent_loss.item()]
            recon_loss_all += [recon_loss.item()]

        history['loss'] += [np.array(latent_loss_all) + np.array(recon_loss_all)]
        if np.isnan(history['loss'][-1].mean()):
            raise ValueError
        # print([x.mean() / 128 for x in history['loss']])

    with torch.no_grad():
        for test_elbo_comb in [[True, False], [False, True], [True, True]]:
            for dt_str, dl in zip(['train'], [train_loader, test_loader]):
                topic_prop_test = []
                for i, x in enumerate(dl):
                    topic_prop_test += [model.get_latent([y.to(device) for y in x], elbo_bool=test_elbo_comb)]
                res = np.concatenate(topic_prop_test)
                np.savetxt(os.path.join(
                    output_dir,
                    file_sub + "_".join([str(x) for x in test_elbo_comb]) + \
                    "_" + dt_str + "_latent.csv"),
                    res, delimiter=",")
