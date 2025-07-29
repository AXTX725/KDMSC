import scipy.io
import numpy as np
from sklearn.metrics import cohen_kappa_score


def calculate_metrics(true_labels, pred_labels):
    '''
    Calculate classification performance metrics including overall accuracy,
    per-class accuracy, and Cohen's kappa score.

    Parameters:
    - true_labels: ground truth labels (with 0 as ignored class)
    - pred_labels: predicted labels

    Returns:
    - overall_accuracy: proportion of correctly predicted non-zero labels
    - class_accuracy: accuracy for each class (index corresponds to class label)
    - kappa: Cohen's kappa coefficient measuring agreement between prediction and ground truth
    '''

    total_samples = np.sum(true_labels != 0)
    correct_samples = np.sum((true_labels == pred_labels) & (true_labels != 0))
    overall_accuracy = correct_samples / total_samples

    class_accuracy = np.zeros(np.max(true_labels) + 1)
    for i in range(1, np.max(true_labels) + 1):
        class_samples = np.sum(true_labels == i)
        class_correct_samples = np.sum((true_labels == i) & (pred_labels == i))
        class_accuracy[i] = class_correct_samples / class_samples

    true_labels_flatten = true_labels.flatten()
    pred_labels_flatten = pred_labels.flatten()

    nonzero_indices = np.where((true_labels_flatten != 0) & (pred_labels_flatten != 0))

    true_labels_nonzero = true_labels_flatten[nonzero_indices]
    pred_labels_nonzero = pred_labels_flatten[nonzero_indices]

    kappa = cohen_kappa_score(true_labels_nonzero, pred_labels_nonzero)

    return overall_accuracy, class_accuracy, kappa


true_labels = scipy.io.loadmat('data/PaviaU/PaviaU_gt.mat')['paviaU_gt']  # 真实标签


total_oa = 0
total_aa = np.zeros(np.max(true_labels) + 1)
total_kappa = 0
for i in range(10):
    pred_labels = scipy.io.loadmat(f'prediction/PaviaU/10/50_overall_skip_2_SGConv_l1_clip/{i}.mat')['pred'] + 1
    oa, aa, kappa = calculate_metrics(true_labels, pred_labels)
    total_oa += oa
    total_aa += aa
    total_kappa += kappa


average_oa = total_oa / 10
average_aa = total_aa / 10
average_kappa = total_kappa / 10


print("Average Overall Accuracy（OA）： {:.2f}%".format(average_oa * 100))
print("Average Accuracy（AA）：")
for i in range(1, np.max(true_labels) + 1):
    print("Accuracy for class ", i, "： {:.2f}%".format(average_aa[i] * 100))
print("AA： {:.2f}%".format(np.sum(average_aa) / np.max(true_labels) * 100))
print("Average Kappa coefficient： {:.2f}".format(average_kappa))