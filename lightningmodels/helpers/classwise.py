import torch


def classwise(y_hat, y, metric):

    assert y_hat.shape == y.shape

    results = torch.empty(y_hat.shape[1],
                          dtype=y_hat.dtype,
                          device=y_hat.device)

    for i in torch.tensor(range(y_hat.shape[1]),
                          dtype=torch.long,
                          device=y_hat.device):
        y_hat_class = y_hat.index_select(1, i)
        y_class = y.index_select(1, i)
        results[i] = metric(y_hat_class, y_class)

    return results
