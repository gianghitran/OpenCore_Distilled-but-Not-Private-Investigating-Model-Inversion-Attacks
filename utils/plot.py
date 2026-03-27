import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wandb

def select_n_per_class(X, y, n=1000):
    import pandas as pd
    df = pd.DataFrame(X)
    df['label'] = y
    df_selected = df.groupby('label', group_keys=False).head(n)
    X_selected = df_selected.drop(columns=['label']).values
    y_selected = df_selected['label'].values
    return X_selected, y_selected

def plot_tsne(X_orig, y_orig, X_inv, y_inv, class_names, title, save_path, prefix=""):
    # Lấy 1000 sample đầu tiên mỗi class cho cả original và inverted
    X_orig, y_orig = select_n_per_class(X_orig, y_orig, n=1000)
    X_inv, y_inv = select_n_per_class(X_inv, y_inv, n=1000)

    X_all = np.vstack([X_orig, X_inv])
    y_all = np.concatenate([y_orig, y_inv])
    domain = np.array([0]*len(X_orig) + [1]*len(X_inv))  # 0: original, 1: inverted

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_all)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', len(class_names))

    handles = []
    labels = []

    # Vẽ tất cả dấu . (original) trước
    for i, cls in enumerate(class_names):
        idx_orig = np.where((y_all == cls) & (domain == 0))[0]
        sc = plt.scatter(X_embedded[idx_orig, 0], X_embedded[idx_orig, 1],
                         c=[colors(i)], marker='.', label=f'{int(cls)} (orig)', alpha=0.7, edgecolors='none')
        handles.append(sc)
        labels.append(f'{int(cls)} (orig)')

    # Sau đó vẽ tất cả dấu + (inverted)
    for i, cls in enumerate(class_names):
        idx_inv = np.where((y_all == cls) & (domain == 1))[0]
        sc = plt.scatter(X_embedded[idx_inv, 0], X_embedded[idx_inv, 1],
                         c=[colors(i)], marker='+', label=f'{int(cls)} (inv)', alpha=0.7)
        handles.append(sc)
        labels.append(f'{int(cls)} (inv)')

    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(handles, labels, markerscale=1.5, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    wandb.log({f"{prefix}t-SNE": wandb.Image(save_path)})