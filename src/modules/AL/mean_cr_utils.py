import pickle

import numpy as np


def compute_meanclassification_report(report_paths, save=None, verbose=0, store_dict=False):
    c_dicts = []
    for path in report_paths:
        with open(path, 'rb') as f:
            c_dicts.append(pickle.load(f))
        c_dicts.append(np.load(path, allow_pickle=True))
    structure, means, stds = extract_mean_report(c_dicts)
    mean_dict = build_mean_dict(structure, means, stds)
    pretty_mean = pretty_print(mean_dict)

    if verbose:
        print(pretty_mean)
    if not save is None:
        with open(save, 'w') as f:
            f.write(pretty_mean)
        if verbose:
            print('Save Location:', save)
        if store_dict:
            with open(save.replace('.txt', '.pickle'), 'wb') as f:
                pickle.dump(mean_dict, f)


def extract_mean_report(c_dicts):
    structure, values, sflag = [], [], True
    for cr in c_dicts:
        tmp_vals = []
        for k1, v1 in cr.items():
            if type(v1) is dict:
                for k2, v2 in v1.items():
                    tmp_vals.append(v2)
                    if sflag:
                        structure.append([k1, k2])
            else:
                tmp_vals.append(v1)
                if sflag:
                    structure.append(k1)
        values.append(np.array(tmp_vals))
        sflag = False
    values = np.array(values)
    means, stds = np.mean(values, axis=0), np.std(values, axis=0)
    return structure, means, stds


def build_mean_dict(structure, means, stds):
    d = {}
    for k, m, s in zip(structure, means, stds):
        if type(k) is str:
            d[k] = '%.4f + %.4f' % (m, s)
        else:
            if k[0] not in d.keys():
                d[k[0]] = {}
            for k2 in k[1:]:
                d[k[0]][k2] = '%.4f + %.4f' % (m, s)
    return d


def pretty_print(mean_dict):
    longest_last_line_heading = 'weighted avg'
    headers = ["precision", "recall", "f1-score", "support"]
    target_names = list(mean_dict.keys())[:-3]
    val_len = len(mean_dict['accuracy'])
    headers = [' ' * (val_len - len(h)) + h for h in headers]

    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading),
                len(mean_dict['accuracy']))
    head_fmt = '{:>{width}s}\t' + '{:>9}\t\t' * 2 + '{:>9}\t' + '{:>9}'
    report = head_fmt.format('', *headers, width=width)
    report += '\n\n'
    row_fmt = '{:>{width}s}\t' + '{:>{width}s}\t\t' * 2 + '{:>{width}s}\t'
    for r_key in list(mean_dict.keys())[:-3]:
        row = [r_key] + [v for v in mean_dict[r_key].values()]
        row[-1] = row[-1].split('.')[0]
        tmp = row_fmt.format(*row[:-1], width=width) + ' ' * \
            (len(headers[-1]) - len(row[-1])) + row[-1] + '\n'
        report += tmp
    report += '\n'
    for r_key in list(mean_dict.keys())[-3:]:
        if r_key == 'accuracy':
            row = ['', '', mean_dict['accuracy']] + \
                [mean_dict['macro avg']['support']]
        else:
            row = [v for v in mean_dict[r_key].values()]
        row[-1] = row[-1].split('.')[0]
        tmp = row_fmt.format(r_key, *row, width=width) + ' ' * \
            (len(headers[-1]) - len(row[-1])) + row[-1] + '\n'
        report += tmp
    return report
