from ot import gromov_wasserstein2

def l2_distance(X, Y):
    return torch.sum((X - Y)**2, 1)

def geometry_score(X, Y):
    if torch.cuda.is_available():
        rlts1 = rlts(X.data.cpu().numpy(), n=mb_size)
        rlts2 = rlts(Y.data.cpu().numpy(), n=mb_size)
    else:
        rlts1 = rlts(X.data.numpy(), n=mb_size)
        rlts2 = rlts(Y.data.numpy(), n=mb_size)
    return Variable(Tensor(geom_score(rlts1, rlts2)),
                    requires_grad=False)

def gromov_wasserstein_distance(X, Y):
    import concurrent.futures
    gw_dist = np.zeros(mb_size)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in executor.map(range(mb_size)):
            C1 = sp.spatial.distance.cdist(X[i,:].reshape(28,28).data.cpu().numpy(), X[i,:].reshape(28,28).data.cpu().numpy()) #Convert data back to an image from one hot encoding with size 28x28
            C2 = sp.spatial.distance.cdist(Y[i,:].reshape(28,28).data.cpu().numpy(), Y[i,:].reshape(28,28).data.cpu().numpy())
            C1 /= C1.max()
            C2 /= C2.max()
            p = unif(28)
            q = unif(28)
            gw_dist[i] = gromov_wasserstein2(C1, C2, p, q, loss_fun='square_loss', epsilon=5e-4)
            return Variable(Tensor(gw_dist), requires_grad=False)

def topological_entropy(prior_distribution, posterior_distribution):
    """
    Args:
        prior: a Conv2d
        poster: a Conv2d
    Return:
        score
    """
    if torch.cuda.is_available():
        rlts1 = rlts(X.data.cpu().numpy(), n=mb_size)
        rlts2 = rlts(Y.data.cpu().numpy(), n=mb_size)
    else:
        rlts1 = rlts(X.data.numpy(), n=mb_size)
        rlts2 = rlts(Y.data.numpy(), n=mb_size)
    return Variable(Tensor(geom_score(rlts1, rlts2)),
                    requires_grad=False).sum()
