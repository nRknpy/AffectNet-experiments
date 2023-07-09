from typing import Literal

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from umap.umap_ import UMAP
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


labeled_valence = {
    0: 'valence < -0.5',
    1: '-0.5 <= valence <= 0.5',
    2: '0.5 < valence',
}


def head_outputs(model, dataset, device):
    features = []
    labels = []
    for img, label in tqdm(dataset):
        if isinstance(img, tuple):
            img = img[0]
        with torch.no_grad():
            feature = model(img.unsqueeze(0).to(device)).logits
        features.append(feature.cpu())
        labels.append(label)
    return torch.stack(features).squeeze(), torch.tensor(labels)


def CLS_tokens(model, dataset, device):
    tokens = []
    labels = []
    for img, label in tqdm(dataset):
        if isinstance(img, tuple):
            img = img[0]
        with torch.no_grad():
            token = model(img.unsqueeze(0).to(device),
                          output_hidden_states=True).hidden_states[-1][0, 0, :]
        tokens.append(token.cpu())
        labels.append(label)
    return torch.stack(tokens).squeeze(), torch.tensor(labels)


def hidden_tokens(model, dataset, device):
    tokens_list = [[] for _ in range(12)]
    labels = []
    for img, label in tqdm(dataset):
        if isinstance(img, tuple):
            img = img[0]
        with torch.no_grad():
            tokens = model(img.unsqueeze(0).to(device),
                           output_hidden_states=True).hidden_states
        for i in range(1, 13):
            tokens_list[i-1].append(tokens[i][0, 0, :].cpu())
        labels.append(label)
    for i in range(len(tokens_list)):
        tokens_list[i] = torch.stack(tokens_list[i])
    return tokens_list, torch.tensor(labels)


def plot_tokens_category(tokens, labels, n_neighbors, id2label, random_seed, method: Literal['umap', 'mds'] = 'umap'):
    if method == 'umap':
        compress = UMAP(n_neighbors=n_neighbors, random_state=random_seed)
    elif method == 'mds':
        compress = MDS(n_components=2, random_state=random_seed, n_init=2)
        tokens = F.normalize(tokens, dim=0)
    zs = compress.fit_transform(tokens.numpy())
    ys = labels.numpy()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('feature-1')
    ax.set_ylabel('feature-2')
    cmap = cm.get_cmap('gist_ncar')
    ax.set_box_aspect(1)

    label2point = {}
    print('plotting tokens...')
    for x, y in tqdm(zip(zs, ys)):
        mp = ax.scatter(x[0], x[1],
                        alpha=1,
                        label=id2label[y],
                        c=y,
                        cmap=cmap,
                        vmin=0,
                        vmax=len(set(ys)),
                        s=3,)
        label2point[id2label[y]] = mp
    labels, handles = zip(*sorted(label2point.items()))
    # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.5))
    legend = ax.legend(handles, labels, loc='upper left',
                       bbox_to_anchor=(1, 0.5))
    return fig, legend


def plot_tokens_continuous(tokens, targets, n_neighbors, random_seed, method: Literal['umap', 'mds'] = 'umap'):
    if method == 'umap':
        compress = UMAP(n_neighbors=n_neighbors, random_state=random_seed)
    elif method == 'mds':
        compress = MDS(n_components=2, random_state=random_seed, n_init=2)
        tokens = F.normalize(tokens, dim=0)
    zs = np.array(compress.fit_transform(tokens.numpy()))
    x = zs[:, 0]
    y = zs[:, 1]
    z = targets.numpy()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('feature-1')
    ax.set_ylabel('feature-2')
    ax.set_box_aspect(1)

    mp = ax.scatter(x, y,
                    alpha=1,
                    c=z,
                    cmap='Oranges',
                    vmin=-1,
                    vmax=1,
                    s=3)
    fig.colorbar(mp, ax=ax)
    return fig


def plot_hidden_tokens_category(tokens_list, labels, n_neighbors, id2label, random_seed, method: Literal['umap', 'mds'] = 'umap'):
    ys = labels.numpy()
    vmax = len(set(ys))
    fig = plt.figure(figsize=(100, 8))
    fig.tight_layout()
    cmap = cm.get_cmap('gist_ncar')
    for i in tqdm(range(len(tokens_list))):
        ax = fig.add_subplot(1, len(tokens_list), i+1)
        ax.set_box_aspect(1)
        ax.set_title(f'{i+1}', fontsize=100)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        if method == 'umap':
            compress = UMAP(n_neighbors=n_neighbors, random_state=random_seed)
        elif method == 'mds':
            compress = MDS(n_components=2, random_state=random_seed, n_init=2)
            tokens_list[i] = F.normalize(tokens_list[i], dim=0)
        zs = compress.fit_transform(tokens_list[i].numpy())
        label2point = {}
        for x, y in zip(zs, ys):
            mp = ax.scatter(x[0], x[1],
                            alpha=1,
                            label=id2label[y],
                            c=y,
                            cmap=cmap,
                            vmin=0,
                            vmax=vmax,
                            s=20,)
            label2point[id2label[y]] = mp
        labels_, handles = zip(*sorted(label2point.items()))
    # legend = fig.legend(
    #     handles, labels_, loc='upper center', ncol=8, fontsize=100)
    return fig
