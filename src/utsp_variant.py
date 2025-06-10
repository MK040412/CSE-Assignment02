import numpy as np

def compute_heatmap(coords, tau=10.0):
    n = len(coords)
    D = np.sqrt(((np.array(coords)[:,None,:] - np.array(coords)[None,:,:])**2).sum(axis=2))
    W = np.exp(-D / tau)
    # remove self loops
    np.fill_diagonal(W, 0)
    # column softmax
    expW = np.exp(W - W.max(axis=0, keepdims=True))
    T = expW / expW.sum(axis=0, keepdims=True)
    # row normalization (Sinkhorn style)
    T = T / T.sum(axis=1, keepdims=True)
    return T

def utsp_variant_tour(coords):
    T = compute_heatmap(coords)
    n = len(coords)
    visited = set([0])
    tour = [0]
    while len(tour) < n:
        current = tour[-1]
        # choose next by highest heat value excluding visited
        probs = T[current]
        for idx in np.argsort(-probs):
            if idx not in visited:
                tour.append(idx)
                visited.add(idx)
                break
    tour.append(0)
    return tour
