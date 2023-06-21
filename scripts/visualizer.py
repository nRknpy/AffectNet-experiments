import torch
import numpy as np
from tqdm import tqdm
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


labeled_valence = {
    0: 'valence < -0.5',
    1: '-0.5 <= valence <= 0.5',
    2: '0.5 < valence',
}


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


def plot_tokens_category(tokens, labels, n_neighbors, id2label, random_seed):
    umap = UMAP(n_neighbors=n_neighbors, random_state=random_seed)
    zs = umap.fit_transform(tokens.numpy())
    ys = labels.numpy()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('feature-1')
    ax.set_ylabel('feature-2')
    cmap = cm.get_cmap('gist_ncar')

    label2point = {}
    for x, y in zip(zs, ys):
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


def plot_tokens_continuous(tokens, targets, n_neighbors):
    umap = UMAP(n_neighbors=n_neighbors)
    zs = np.array(umap.fit_transform(tokens.numpy()))
    x = zs[:, 0]
    y = zs[:, 1]
    z = targets.numpy()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('feature-1')
    ax.set_ylabel('feature-2')

    mp = ax.scatter(x, y,
                    alpha=1,
                    c=z,
                    cmap='Oranges',
                    vmin=-1,
                    vmax=1,
                    s=3)
    fig.colorbar(mp, ax=ax)
    return fig
