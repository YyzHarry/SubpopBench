import argparse
import numpy as np
from subpopbench.learning import model_selection
from subpopbench.utils import reporting


if __name__ == "__main__":
    """
    Example usage:
    python -u -m subpopbench.scripts.list_top_hparams --input_dir ... --algorithm ERM --dataset Waterbirds
    """
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--algorithm', required=True)
    args = parser.parse_args()

    records = reporting.load_records(args.input_dir)
    print("Total records:", len(records))

    records = reporting.get_grouped_records(records)
    records = records.filter(
        lambda r: r['dataset'] == args.dataset and
                  r['algorithm'] == args.algorithm
    )

    SELECTION_METHODS = [
        # model_selection.OracleMeanAcc,
        # model_selection.OracleWorstAcc,
        # model_selection.ValMeanAcc,
        model_selection.ValWorstAccAttributeYes,
        # model_selection.ValWorstAccAttributeNo,
        # model_selection.ValMeanPrecision,
        # model_selection.ValWorstPrecision,
        # model_selection.ValMeanF1,
        # model_selection.ValWorstF1,
        # model_selection.ValBalancedAcc,
        # model_selection.ValAUROC,
        # model_selection.ValAUPRC,
        # model_selection.ValClassDiff,
    ]

    for selection_method in SELECTION_METHODS:
        print(f'\n\nModel selection: [{selection_method.name}]')

        for group in records:
            print(f"(trial) seed: {group['seed']}")
            best_hparams = selection_method.hparams_accs(group['records'])
            # 'best_hparams' sorted by 'val_acc'
            run_acc, hparam_records = best_hparams[0]
            print(f"\t{run_acc}")
            for r in hparam_records:
                assert(r['hparams'] == hparam_records[0]['hparams'])
            print("\t\thparams:")
            for k, v in sorted(hparam_records[0]['hparams'].items()):
                print('\t\t\t{}: {}'.format(k, v))
            print("\t\toutput_dirs:")
            output_dirs = hparam_records.select('args.output_dir').unique()
            for output_dir in output_dirs:
                print(f"\t\t\t{output_dir}")
