import argparse
import os
import sys
import numpy as np

from subpopbench.dataset import datasets
from subpopbench.learning import algorithms, model_selection
from subpopbench.utils import misc, reporting
from subpopbench.utils.query import Q


def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, f"{mean:.1f} \scriptsize$\\pm{err:.1f}$"
    else:
        return mean, err, f"{mean:.1f} +/- {err:.1f}"


def print_table(table, header_text, row_labels, col_labels, colwidth=10, latex=True):
    """Pretty-print a 2D array of dataset, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}" for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")


def print_results_tables(records, selection_method, latex):
    # common selection for most datasets
    grouped_records = reporting.get_grouped_records(records).map(
        lambda group: {**group,
                       "sweep_acc": selection_method.sweep_acc(group["records"]),
                       "sweep_acc_worst": selection_method.sweep_acc_worst(group["records"]),
                       "sweep_precision": selection_method.sweep_precision(group["records"]),
                       "sweep_precision_worst": selection_method.sweep_precision_worst(group["records"]),
                       "sweep_f1": selection_method.sweep_f1(group["records"]),
                       "sweep_f1_worst": selection_method.sweep_f1_worst(group["records"]),
                       "sweep_acc_adjusted": selection_method.sweep_acc_adjusted(group["records"]),
                       "sweep_acc_balanced": selection_method.sweep_acc_balanced(group["records"]),
                       "sweep_auroc": selection_method.sweep_auroc(group["records"]),
                       "sweep_worst_auroc": selection_method.sweep_worst_auroc(group["records"]),
                       "sweep_ece": selection_method.sweep_ece(group["records"])}
    ).filter(lambda g: g["sweep_acc"] is not None)

    # AUC selection for certain datasets
    auc_grouped_records = reporting.get_grouped_records(records).map(
        lambda group: {**group,
                       "sweep_acc": model_selection.ValAUROC.sweep_acc(group["records"]),
                       "sweep_acc_worst": model_selection.ValAUROC.sweep_acc_worst(group["records"]),
                       "sweep_precision": model_selection.ValAUROC.sweep_precision(group["records"]),
                       "sweep_precision_worst": model_selection.ValAUROC.sweep_precision_worst(group["records"]),
                       "sweep_f1": model_selection.ValAUROC.sweep_f1(group["records"]),
                       "sweep_f1_worst": model_selection.ValAUROC.sweep_f1_worst(group["records"]),
                       "sweep_acc_adjusted": model_selection.ValAUROC.sweep_acc_adjusted(group["records"]),
                       "sweep_acc_balanced": model_selection.ValAUROC.sweep_acc_balanced(group["records"]),
                       "sweep_auroc": model_selection.ValAUROC.sweep_auroc(group["records"]),
                       "sweep_worst_auroc": selection_method.sweep_worst_auroc(group["records"]),
                       "sweep_ece": model_selection.ValAUROC.sweep_ece(group["records"])}
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
                 [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    # print a summary table for each dataset
    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))

        metrics = ["Avg", "Worst", "AvgPrec", "WorstPrec", "AvgF1", "WorstF1", "Adjusted", "Balanced", "AUROC", "ECE"]
        metrics_sweep = ["sweep_acc", "sweep_acc_worst", "sweep_precision", "sweep_precision_worst", "sweep_f1",
                         "sweep_f1_worst", "sweep_acc_adjusted", "sweep_acc_balanced", "sweep_auroc", "sweep_ece"]
        table = [[None for _ in metrics] for _ in alg_names]

        curr_records = auc_grouped_records if dataset in ["MIMICNotes", "CXRMultisite"] else grouped_records
        for i, algorithm in enumerate(alg_names):
            for j, sweep_name in enumerate(metrics_sweep):
                trial_accs = (curr_records.filter_equals("dataset, algorithm", (dataset, algorithm)).select(sweep_name))
                _, _, table[i][j] = format_mean(trial_accs, latex)

        col_labels = ["Algorithm"] + metrics
        header_text = (f"Dataset: {dataset}, "
                       f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels), colwidth=25 if latex else 12, latex=latex)

    # print a overall "worst-case" table
    if latex:
        print()
        print("\\subsubsection{Overall}")

    table = [[None for _ in [*dataset_names, "Worst"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            curr_records = auc_grouped_records if dataset in ["MIMICNotes", "CXRMultisite"] else grouped_records
            if dataset == "Living17":
                dset_metric = "sweep_acc"
            elif dataset in ["MIMICNotes", "CXRMultisite"]:
                dset_metric = "sweep_worst_auroc"
            else:
                dset_metric = "sweep_acc_worst"
            trial_averages = (
                curr_records.filter_equals(
                    "algorithm, dataset", (algorithm, dataset)
                ).group("seed").map(
                    lambda trial_seed, group: group.select(dset_metric).mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = f"{sum(means) / len(means):.1f}"

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Worst-case accuracy, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25 if latex else 12, latex=latex)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full SubpopBench results}")
        print("% Total records:", len(records))
    else:
        print(f"Total records: [{len(records)}]")

    SELECTION_METHODS = [
        model_selection.OracleWorstAcc,
        model_selection.ValWorstAccAttributeYes,
        model_selection.ValWorstAccAttributeNo,
    ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args.latex)

    if args.latex:
        print("\\end{document}")
