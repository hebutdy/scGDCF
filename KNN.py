import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
topk = 10

def construct_graph(features, label, method):
    fname = 'E:/data/mouse_graph.txt'
    num = len(label)
    dist = None
    # Several methods of calculating the similarity relationship between samples i and j (similarity matrix Sij)
    if method == 'heat':
        dist = -0.5 * pair(features, metric='manhattan') ** 2
        dist = np.exp(dist)

    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)

    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    # elif method == 'cos':
    #     dist = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(features.T))

    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    print('error rate: {}'.format(counter / (num * topk)))

# Para = [batch_size, lr, epoch, n_cluster, n_init]
Para = [1024, 1e-3, 200, 16, 20]
method = ['heat', 'cos', 'ncos', 'p']
number = pd.read_csv("E:/data/mouse.csv")
mtab_features = number.loc[:, number.columns[1:]]
print(mtab_features.shape)
print(mtab_features)
label = pd.read_csv("E:/data/mouse_label.csv")
mtab_labels = label.loc[:, "true_label"]
print(mtab_labels.shape)
construct_graph(mtab_features, mtab_labels, method[3])
