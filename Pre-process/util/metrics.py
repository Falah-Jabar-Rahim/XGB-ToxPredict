import numpy as np
from sklearn.metrics import f1_score


def compute_ece(y_true, y_prob, n_bins):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    ece = 0.0
    for i in range(1, n_bins + 1):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            bin_confidence = np.mean(y_prob[bin_mask])
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_error = abs(bin_confidence - bin_accuracy)
            ece += (bin_size / len(y_true)) * bin_error

    return ece


def F1_Score(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro')
    return np.array(f1)
