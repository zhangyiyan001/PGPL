# -*- coding:utf-8 -*-
"""
Author: Yiyan Zhang
Date: September 22, 2022
"""
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import os
import torch
import time
import numpy as np
from module import AA_andEachClassAccuracy, dis_groundtruth

def _get_target_names(dataset: str):
    if dataset == 'Houston':
        return ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Tree',
                'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                'Railway', 'Parking lot 1', 'Parking lot 2', 'Tennis court', 'Running track']
    if dataset == 'Trento':
        return ['Apple trees', 'Buildings', 'Ground', 'Wood', 'Vineyard', 'Roads']
    if dataset == 'MUUFL':
        return ['Trees', 'Mostly grass', 'Mixed ground surface', 'Dirt and sand', 'Road', 'Water',
                'Building shadow', 'Building', 'Sidewalk', 'Yellow curb', 'Cloth panels']
    raise ValueError(f'Unsupported dataset: {dataset}')

def _compute_calibration(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    """Compute ECE calibration bins and Brier score for multi-class.

    Returns: ece, brier, bins(dict)
    """
    preds = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(probs)), preds]
    correct = (preds == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = []
    bin_conf = []
    bin_count = []
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        m = np.sum(mask)
        bin_count.append(int(m))
        if m > 0:
            acc = float(np.mean(correct[mask]))
            conf = float(np.mean(confidences[mask]))
            ece += (m / N) * abs(acc - conf)
        else:
            acc, conf = 0.0, 0.0
        bin_acc.append(acc)
        bin_conf.append(conf)

    # Brier score (multi-class)
    n_class = probs.shape[1]
    one_hot = np.eye(n_class, dtype=np.float32)[y_true]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    calib = {
        'bin_edges': bin_edges.tolist(),
        'bin_acc': bin_acc,
        'bin_conf': bin_conf,
        'bin_count': bin_count,
    }
    return float(ece), brier, calib


def evaluate_model_metrics(model, test_iter, dataset, device):
    """
    Evaluate the model on a test set and return the main metrics.
    """
    was_training = model.training
    model.eval()

    tick1 = time.time()
    y_test = []
    y_pred = []
    prob_chunks = []
    with torch.no_grad():
        for X1, X2, y in test_iter:
            X1 = X1.to(device)
            X2 = X2.to(device)
            y = y.to(device)
            outputs = model(X1, X2)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(np.argmax(probs, axis=1))
            prob_chunks.append(probs)
            y_test.extend(y.cpu().numpy())
    tick2 = time.time()

    if was_training:
        model.train()

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    probs_all = np.vstack(prob_chunks) if len(prob_chunks) > 0 else None
    target_names = _get_target_names(dataset)

    classification = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    # Calibration metrics
    ece, brier, calib = (0.0, 0.0, None)
    if probs_all is not None and len(y_test) == probs_all.shape[0]:
        ece, brier, calib = _compute_calibration(probs_all, y_test, n_bins=15)

    return {
        'oa': oa,
        'aa': aa,
        'kappa': kappa,
        'classification_report': classification,
        'confusion_matrix': confusion,
        'each_class_accuracy': each_acc,
        'eval_time': tick2 - tick1,
        'test_time': tick2 - tick1,
        'y_true': y_test,
        'y_pred': y_pred,
        'target_names': target_names,
        'ece': ece,
        'brier': brier,
        'calibration': calib
    }

def log_metrics_to_file(metrics, dataset, stage_name=None, hyperparams=None):
    """
    Write evaluation metrics to the result file with an optional stage label.
    """
    prefix = f'[{stage_name}] ' if stage_name else ''
    file_name = "./results/{}/{}.txt".format(dataset, dataset)
    with open(file_name, 'a') as x_file:
        x_file.write('\n**************************************************************************************\n')
        x_file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        x_file.write('\n')
        x_file.write(f'{prefix}{metrics["oa"]} Overall accuracy (%)')
        x_file.write('\n')
        x_file.write(f'{prefix}{metrics["aa"]} Average accuracy (%)')
        x_file.write('\n')
        x_file.write(f'{prefix}{metrics["kappa"]} Kappa accuracy (%)')
        x_file.write('\n')
        if hyperparams:
            x_file.write('\n')
            x_file.write('Hyperparameters:\n')
            x_file.write(f'{hyperparams}\n')
        x_file.write('\n')
        x_file.write('Training Time {}s'.format(metrics['eval_time']))
        x_file.write('\n')
        x_file.write('Testing Time {}s'.format(metrics['test_time']))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(metrics['classification_report']))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('mean_OA is: ' + str(np.mean(metrics['oa'])))
        x_file.write('\n')
        x_file.write('mean_AA is: ' + str(np.mean(metrics['aa'])))
        x_file.write('\n')
        x_file.write('mean_KAPPA  is: ' + str(np.mean(metrics['kappa'])))
        x_file.write('\n')
        if 'ece' in metrics:
            x_file.write(f"ECE: {metrics['ece']:.6f}\n")
        if 'brier' in metrics:
            x_file.write(f"Brier: {metrics['brier']:.6f}\n")
        x_file.write('\n**************************************************************************************\n')
        x_file.write('\n')
        x_file.write('\n')
    # Save a reliability diagram when calibration data is available.
    try:
        calib = metrics.get('calibration')
        if calib is not None:
            import matplotlib.pyplot as plt
            os.makedirs(f'./results/{dataset}', exist_ok=True)
            fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
            bins = calib['bin_edges']
            acc = calib['bin_acc']
            conf = calib['bin_conf']
            centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
            ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
            ax.plot(centers, acc, marker='o', label='Accuracy')
            ax.plot(centers, conf, marker='s', label='Confidence')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy / Confidence')
            title = f'Reliability Diagram'
            if stage_name:
                title += f' ({stage_name})'
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='lower right')
            safe_stage = stage_name.replace('@', '_').replace(':', '_') if stage_name else 'Eval'
            fig.savefig(f'./results/{dataset}/reliability_{safe_stage}.png', bbox_inches='tight')
            plt.close(fig)
    except Exception:
        pass

def result(test_iter, dataset, device, net, stage_name=None, hyperparams=None):
    net.load_state_dict(torch.load('./models/' + dataset + '.pt'))
    print('\n***Start  Testing***\n')
    metrics = evaluate_model_metrics(net, test_iter, dataset, device)
    log_metrics_to_file(metrics, dataset, stage_name=stage_name, hyperparams=hyperparams)
    stage_info = f' Stage: {stage_name}' if stage_name else ''
    print(f"Evaluation complete{stage_info} | OA={metrics['oa']:.4f}, AA={metrics['aa']:.4f}, Kappa={metrics['kappa']:.4f}")


