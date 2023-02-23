import numpy as np
from subpopbench.utils.misc import safe_load


class SelectionMethod:
    """
    Abstract class whose subclasses implement strategies for model selection across hparams & steps
    """
    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(cls, run_records):
        """
        Given records from a run, return a {val_acc, test_acc, ...} dict representing
        the best val-acc, corresponding test-acc and other test metrics for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(cls, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed').map(
            lambda _, run_records: (
                cls.run_acc(run_records),
                run_records
            )
        ).filter(lambda x: x[0] is not None).sorted(key=lambda x: x[0]['val_acc'])[::-1])

    @classmethod
    def sweep_acc(cls, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

    @classmethod
    def sweep_acc_worst(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc_worst']
        else:
            return None

    @classmethod
    def sweep_precision(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_precision']
        else:
            return None

    @classmethod
    def sweep_precision_worst(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_precision_worst']
        else:
            return None

    @classmethod
    def sweep_f1(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_f1']
        else:
            return None

    @classmethod
    def sweep_f1_worst(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_f1_worst']
        else:
            return None

    @classmethod
    def sweep_acc_adjusted(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc_adjusted']
        else:
            return None

    @classmethod
    def sweep_acc_balanced(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc_balanced']
        else:
            return None

    @classmethod
    def sweep_auroc(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_auroc']
        else:
            return None

    @classmethod
    def sweep_worst_auroc(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_worst_auroc']
        else:
            return None

    @classmethod
    def sweep_auprc(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_auprc']
        else:
            return None

    @classmethod
    def sweep_ece(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_ece']
        else:
            return None

    @classmethod
    def get_test_split(cls, record):
        if record['args']['dataset'] == 'ImagenetBG':
            return 'mixed_rand'
        elif record['args']['dataset'] == 'Living17':
            return 'zs'
        else:
            return 'te'


class OracleMeanAcc(SelectionMethod):
    """Picks argmax(mean(grp_test_acc for grp in all_groups))"""
    name = "test set mean accuracy (oracle)"

    @classmethod
    def _step_acc(cls, record):
        """Given a single record, return a {val_acc, test_acc, ...} dict."""
        te = cls.get_test_split(record)
        return {'val_acc': record[te]['overall']['accuracy'],
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}

    @classmethod
    def run_acc(cls, run_records):
        return run_records.map(cls._step_acc).argmax('test_acc')


class OracleWorstAcc(SelectionMethod):
    """Picks argmax(min(grp_test_acc for grp in all_groups))"""
    name = "test set worst accuracy (oracle)"

    @classmethod
    def _step_acc(cls, record):
        """Given a single record, return a {val_acc, test_acc, ...} dict."""
        te = cls.get_test_split(record)
        return {'val_acc': record[te]['min_group']['accuracy'],
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}

    @classmethod
    def run_acc(cls, run_records):
        return run_records.map(cls._step_acc).argmax('test_acc_worst')


class ValMeanAcc(SelectionMethod):
    # attribute agnostic
    name = "validation set mean accuracy"

    @classmethod
    def _step_acc(cls, record):
        te = cls.get_test_split(record)
        return {'val_acc': record['va']['overall']['accuracy'],
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}

    @classmethod
    def run_acc(cls, run_records):
        if not len(run_records):
            return None
        return run_records.map(cls._step_acc).argmax('val_acc')


class ValWorstAccAttributeYes(ValMeanAcc):
    """Picks argmax(min(grp_val_acc for grp in all_groups))"""
    name = "validation set worst accuracy (with attributes)"

    @classmethod
    def _step_acc(cls, record):
        te = cls.get_test_split(record)
        return {'val_acc': record['va']['min_group']['accuracy'],
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}


class ValWorstAccAttributeNo(ValMeanAcc):
    """Picks argmax(min(grp_val_acc for grp in all_groups))"""
    name = "validation set worst accuracy (without attributes)"

    @classmethod
    def _step_acc(cls, record):
        # class becomes the minimum group
        te = cls.get_test_split(record)
        return {'val_acc': np.min([record['va']['per_class'][i]['recall'] for i in record['va']['per_class']]),
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}


class ValMeanPrecision(ValMeanAcc):
    """Picks argmax(mean(cls_val_precision for cls in all_classes))"""
    name = "validation set mean precision"

    @classmethod
    def _step_acc(cls, record):
        te = cls.get_test_split(record)
        return {'val_acc': record['va']['overall']['macro_avg']['precision'],
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}


class ValWorstPrecision(ValMeanAcc):
    """Picks argmax(min(cls_val_precision for cls in all_classes))"""
    name = "validation set worst precision"

    @classmethod
    def _step_acc(cls, record):
        te = cls.get_test_split(record)
        return {'val_acc': np.min([record['va']['per_class'][i]['precision'] for i in record['va']['per_class']]),
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}


class ValMeanF1(ValMeanAcc):
    """Picks argmax(mean(cls_val_f1 for cls in all_classes))"""
    name = "validation set mean f1-score"

    @classmethod
    def _step_acc(cls, record):
        te = cls.get_test_split(record)
        return {'val_acc': record['va']['overall']['macro_avg']['f1-score'],
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}


class ValWorstF1(ValMeanAcc):
    """Picks argmax(min(cls_val_f1 for cls in all_classes))"""
    name = "validation set worst f1-score"

    @classmethod
    def _step_acc(cls, record):
        te = cls.get_test_split(record)
        return {'val_acc': np.min([record['va']['per_class'][i]['f1-score'] for i in record['va']['per_class']]),
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}


class ValBalancedAcc(ValMeanAcc):
    """Picks argmax(balanced_acc)"""
    name = "validation set class-balanced accuracy (macro recall)"

    @classmethod
    def _step_acc(cls, record):
        te = cls.get_test_split(record)
        return {'val_acc': record['va']['overall']['balanced_acc'],
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}


class ValAUROC(ValMeanAcc):
    """Picks argmax(auroc)"""
    name = "validation set AUROC"

    @classmethod
    def _step_acc(cls, record):
        te = cls.get_test_split(record)
        return {'val_acc': safe_load(record['va']['overall']['AUROC']),
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}


class ValClassDiff(SelectionMethod):
    """
    Minimum class difference as model selection without group annotations
    https://openreview.net/pdf?id=TSqRwmrRiOn
    """
    name = "minimum class difference"

    @classmethod
    def _step_acc(cls, record):
        class_diff = 0.
        per_class_acc = [record['va']['per_class'][i]['recall'] for i in record['va']['per_class']]
        for i in range(len(per_class_acc)):
            for j in range(i+1, len(per_class_acc)):
                class_diff += np.abs(per_class_acc[i] - per_class_acc[j])

        te = cls.get_test_split(record)
        return {'val_acc': class_diff,
                'test_acc': record[te]['overall']['accuracy'],
                'test_acc_worst': record[te]['min_group']['accuracy'],
                'test_precision': record[te]['overall']['macro_avg']['precision'],
                'test_precision_worst': np.min(
                    [record[te]['per_class'][i]['precision'] for i in record[te]['per_class']]),
                'test_f1': record[te]['overall']['macro_avg']['f1-score'],
                'test_f1_worst': np.min([record[te]['per_class'][i]['f1-score'] for i in record[te]['per_class']]),
                'test_acc_adjusted': record[te]['adjusted_accuracy'],
                'test_acc_balanced': record[te]['overall']['balanced_acc'],
                'test_auroc': safe_load(record[te]['overall']['AUROC']),
                'test_worst_auroc': safe_load(record[te]['min_attr']['AUROC']),
                'test_ece': record[te]['overall']['ECE']}

    @classmethod
    def run_acc(cls, run_records):
        if not len(run_records):
            return None
        return run_records.map(cls._step_acc).argmin('val_acc')
