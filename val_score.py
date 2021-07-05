import numpy as np

def mae_score(true, pred):
    score = np.mean(np.abs(true - pred))
    return score


def f1_score(true, pred):
    target = np.where((true > 0.05) & (true < 0.5))

    true = true[target]
    pred = pred[target]

    true = np.where(true < 0.15, 0, 1)
    pred = np.where(pred < 0.15, 0, 1)

    right = np.sum(true * pred == 1)
    precision = right / np.sum(true + 1e-8)
    recall = right / np.sum(pred + 1e-8)

    score = 2 * precision * recall / (precision + recall + 1e-8)

    return score


def mae_over_f1(true, pred):
    mae = mae_score(true, pred)
    f1 = f1_score(true, pred)
    score = mae / (f1 + 1e-8)

    return score


def val_score(pred, targ):

    pred = pred.detach().cpu().numpy()
    targ = targ.detach().cpu().numpy()

    score = mae_over_f1(targ, pred)

    return score