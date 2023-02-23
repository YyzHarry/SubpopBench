import argparse
import copy
import os
import shutil
import numpy as np
import tqdm
import shlex

from subpopbench import command_launchers
from subpopbench.dataset import datasets
from subpopbench.learning import algorithms, model_selection
from subpopbench.utils import reporting


class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, slurm_pre=None):
        self.output_dir = os.path.join(
            args.output_dir,
            f"{args.output_folder_name}{'_attrYes' if args.train_attr == 'yes' else '_attrNo'}",
            f"{train_args['dataset']}_{train_args['algorithm']}"
            f"_hparams{train_args['hparams_seed']}_seed{train_args['seed']}"
        )

        self.train_args = copy.deepcopy(train_args)
        command = ['python', '-m', 'subpopbench.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if slurm_pre:
            self.command_str = f'sbatch {slurm_pre} --wrap "{self.command_str}"'

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
                    self.train_args['algorithm'],
                    self.train_args['hparams_seed'],
                    self.train_args['seed'])
        return f'{self.state}: {self.output_dir} {job_info}'

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        if launcher_fn.__code__.co_argcount > 1:
            launcher_fn(commands, [job.output_dir for job in jobs], max_slurm_jobs=args.max_slurm_jobs)
        else:
            launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def load_best_hparams(all_records, dataset, algo):
    records = all_records.filter(
        lambda r: r['dataset'] == dataset and r['algorithm'] == algo)
    selection_method = model_selection.ValWorstAccAttributeYes

    assert len(records) == 1
    group = records[0]
    sorted_hparams = selection_method.hparams_accs(group['records'])
    run_acc, best_hparam_records = sorted_hparams[0]
    for r in best_hparam_records:
        assert(r['hparams'] == best_hparam_records[0]['hparams'])
    output_dir = best_hparam_records.select('args.output_dir').unique()
    assert len(output_dir) == 1
    hp_seed = output_dir[0][output_dir[0].find('hparams') + len('hparams'):output_dir[0].find('_seed')]
    return int(hp_seed)


def make_args_list(n_trials, dataset_names, algorithms, train_attr, n_hparams_from, n_hparams, steps,
                   image_arch, text_arch, stage1_folder, stage1_algo, output_folder_name, hparams):
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                for hparams_seed in range(n_hparams_from, n_hparams):
                    train_args = {}
                    train_args['dataset'] = dataset
                    train_args['algorithm'] = algorithm
                    train_args['train_attr'] = train_attr
                    train_args['output_folder_name'] = output_folder_name
                    train_args['data_dir'] = args.data_dir
                    train_args['hparams_seed'] = hparams_seed
                    train_args['seed'] = trial_seed
                    if image_arch is not None:
                        train_args['image_arch'] = image_arch
                    if text_arch is not None:
                        train_args['text_arch'] = text_arch
                    if stage1_folder is not None:
                        train_args['stage1_folder'] = stage1_folder
                    if stage1_algo is not None:
                        train_args['stage1_algo'] = stage1_algo
                    if steps is not None:
                        train_args['steps'] = steps
                    if hparams is not None:
                        train_args['hparams'] = hparams
                    args_list.append(train_args)
    return args_list


def make_best_hp_args_list(n_trials, dataset_names, algorithms, train_attr, steps,
                           image_arch, text_arch, stage1_folder, stage1_algo, output_folder_name, hparams):
    all_records = reporting.load_records(os.path.join(args.output_dir, args.input_folder))
    all_records = reporting.get_grouped_records(all_records)
    args_list = []
    for dataset in dataset_names:
        for algorithm in algorithms:
            hparams_seed = load_best_hparams(all_records, dataset, algorithm)
            for trial_seed in range(n_trials):
                train_args = {}
                train_args['dataset'] = dataset
                train_args['algorithm'] = algorithm
                train_args['train_attr'] = train_attr
                train_args['output_folder_name'] = output_folder_name
                train_args['data_dir'] = args.data_dir
                train_args['hparams_seed'] = hparams_seed
                train_args['seed'] = trial_seed
                if image_arch is not None:
                    train_args['image_arch'] = image_arch
                if text_arch is not None:
                    train_args['text_arch'] = text_arch
                if stage1_folder is not None:
                    train_args['stage1_folder'] = stage1_folder
                if stage1_algo is not None:
                    train_args['stage1_algo'] = stage1_algo
                if steps is not None:
                    train_args['steps'] = steps
                if hparams is not None:
                    train_args['hparams'] = hparams
                args_list.append(train_args)
    return args_list


def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)


DATASETS = [d for d in datasets.DATASETS if "Toy" not in d]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    # pass through commands / change here each run
    parser.add_argument('command', choices=['launch', 'delete_incomplete', 'delete_all'])
    parser.add_argument('--command_launcher', type=str, default='multi_gpu')
    parser.add_argument('--output_folder_name', type=str, required=True)
    parser.add_argument('--dataset', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--train_attr', type=str, default='yes', choices=['yes', 'no'])
    # sweep with best hparam, different seeds
    parser.add_argument('--best_hp', action='store_true')
    parser.add_argument('--input_folder', type=str, default='vanilla')
    parser.add_argument('--n_trials', type=int, default=3)
    # optional usage
    parser.add_argument('--image_arch', default=None, choices=[
        'resnet_sup_in1k', 'resnet_sup_in21k', 'resnet_simclr_in1k', 'resnet_barlow_in1k', 'resnet_dino_in1k',
        'vit_sup_in1k', 'vit_sup_in21k', 'vit_clip_oai', 'vit_clip_laion', 'vit_sup_swag', 'vit_dino_in1k'])
    parser.add_argument('--text_arch', default=None, choices=[
        'bert-base-uncased', 'gpt2', 'xlm-roberta-base', 'allenai/scibert_scivocab_uncased', 'distilbert-base-uncased'])
    parser.add_argument('--stage1_folder', type=str, default=None)
    parser.add_argument('--stage1_algo', type=str, default=None)
    # default fixed
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--skip_confirmation', action='store_true')
    # slurm launcher
    parser.add_argument('--slurm_pre', type=str, default=None)
    parser.add_argument('--max_slurm_jobs', type=int, default=15)
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=1,
        dataset_names=args.dataset,
        algorithms=args.algorithms,
        train_attr=args.train_attr,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        image_arch=args.image_arch,
        text_arch=args.text_arch,
        stage1_folder=args.stage1_folder,
        stage1_algo=args.stage1_algo,
        output_folder_name=args.output_folder_name,
        hparams=args.hparams
    ) if not args.best_hp else make_best_hp_args_list(
        n_trials=args.n_trials,
        dataset_names=args.dataset,
        algorithms=args.algorithms,
        train_attr=args.train_attr,
        steps=args.steps,
        image_arch=args.image_arch,
        text_arch=args.text_arch,
        stage1_folder=args.stage1_folder,
        stage1_algo=args.stage1_algo,
        output_folder_name=args.output_folder_name,
        hparams=args.hparams
    )

    jobs = [Job(train_args, args.slurm_pre) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)

    elif args.command == 'delete_all':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE or j.state == Job.DONE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
