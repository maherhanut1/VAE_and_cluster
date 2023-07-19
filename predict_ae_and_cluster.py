import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import argparse

from RES_VAE_2 import VAE

from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--dataset_root", "-dr", help="Dataset root dir", type=str, required=True)

parser.add_argument("--save_dir", "-sd", help="Root dir for saving model and data", type=str, default="/Users/maherhanut/Downloads/CNN-VAE-master/")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=500)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=64)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=128)
parser.add_argument("--ch_multi", '-w', help="Channel width multiplier", type=int, default=8)

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--latent_channels", "-lc", help="Number of channels of the latent space", type=int, default=16)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)

# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--feature_scale", "-fs", help="Feature loss scale", type=float, default=1)
parser.add_argument("--kl_scale", "-ks", help="KL penalty scale", type=float, default=1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint", default=True)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")


# Create dataloaders
# This code assumes there is no pre-defined test/train split and will create one for you
transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

train_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(train_loader)
test_images, _ = next(dataiter)

# Create AE network.
vae_net = VAE(channel_in=test_images.shape[1],
              ch=args.ch_multi,
              latent_channels=args.latent_channels).to(device)


# Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
# If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
# it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
save_file_name = args.model_name + "_" + str(args.image_size)
if args.load_checkpoint:
    checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt",
                            map_location="cpu")
    print("Checkpoint loaded")
    vae_net.load_state_dict(checkpoint['model_state_dict'])
else:
    raise ValueError('load check point must be specified and valid')

# Start training loop
vae_net.eval()
# all_vectors = []
# for i, (images, _) in enumerate(tqdm(train_loader, leave=False)):
#     images = images.to(device)
#     bs, c, h, w = images.shape
#     with torch.cuda.amp.autocast():
#         z = vae_net.encoder(images)
#         z = z.cpu().detach().numpy()[...]
#         all_vectors.append(z)
# all_vectors = np.concatenate([v for v in all_vectors], axis=0)
# np.save('all_vectors_z_flowers_ae_latent_16_test_4_small.npy', all_vectors)


#perform clustring
all_vectors = np.load('all_vectors_z_flowers_ae_latent_16_test_4_small.npy')#[..., 0, 0]
flattened_data = all_vectors.reshape(all_vectors.shape[0], -1)

# Create and fit the clustering model
scaler = StandardScaler()
flattened_data = scaler.fit_transform(flattened_data)
kmeans = KMeans(n_clusters=5, max_iter=10000, tol=1e-36)
kmeans.fit(flattened_data)
# Get the cluster labels for each data point
#cluster_labels = kmeans.labels_

test_set = Datasets.ImageFolder(root='test_set', transform=transform)
test_loader = DataLoader(test_set, batch_size=200, shuffle=False)


all_features = np.zeros((200*5, 16))
all_preds = np.zeros((1000, ))
for i, (images, target) in enumerate(tqdm(test_loader, leave=False)):

    z = vae_net.encoder(images)
    z = z.cpu().detach().numpy()[...]
    s = z.reshape(z.shape[0], -1)
    all_features[i * 200: (1 + i) * 200] = s
    pred = kmeans.predict(s)
    print(f'label is {target[0].item()}')
    for i in range(5):
        print(f'{np.count_nonzero(pred == i)} samples got cluster {i}')


scaler = StandardScaler()
data_std = scaler.fit_transform(all_features)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_std)

for i in range(5):
    plt.scatter(data_pca[200*i:(i+1)*200, 0], data_pca[200*i:(i+1)*200, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: 2D Projection of Data')
plt.show()

import itertools
perms = list(itertools.permutations([0, 1, 2, 3, 4]))
for perm in perms:
    