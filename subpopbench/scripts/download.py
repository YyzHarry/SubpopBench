import argparse
import os
import json
import tarfile
import logging
import gdown
import pandas as pd
import numpy as np
from pathlib import Path
from zipfile import ZipFile
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


def download_datasets(data_path, datasets=['celeba', 'waterbirds', 'civilcomments', 'multinli']):
    os.makedirs(data_path, exist_ok=True)
    dataset_downloaders = {
        'celeba': download_celeba,
        'waterbirds': download_waterbirds,
        'civilcomments': download_civilcomments,
        'multinli': download_multinli,
        'imagenetbg': download_imagenetbg,
        'metashift': download_metashift,
        'nico++': download_nicopp,
        'breeds': download_breeds,
        'cmnist': download_cmnist
    }
    for dataset in datasets:
        if dataset in dataset_downloaders:
            dataset_downloaders[dataset](data_path)
        else:
            no_downloader(dataset)


def no_downloader(dataset):
    print(f"Dataset {dataset} cannot be automatically downloaded. Please check the repo for download instructions.")


def download_civilcomments(data_path):
    logging.info("Downloading CivilComments...")
    civilcomments_dir = os.path.join(data_path, "civilcomments")
    os.makedirs(civilcomments_dir, exist_ok=True)
    download_and_extract(
        "https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/",
        os.path.join(civilcomments_dir, "civilcomments.tar.gz"),
    )


def download_multinli(data_path):
    logging.info("Downloading MultiNLI...")
    multinli_dir = os.path.join(data_path, "multinli")
    glue_dir = os.path.join(multinli_dir, "glue_data/MNLI/")
    os.makedirs(glue_dir, exist_ok=True)
    multinli_tar = os.path.join(glue_dir, "multinli_bert_features.tar.gz")
    download_and_extract(
        "https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz",
        multinli_tar,
    )
    os.makedirs(os.path.join(multinli_dir, "data"), exist_ok=True)
    download_and_extract(
        "https://raw.githubusercontent.com/kohpangwei/group_DRO/master/dataset_metadata/multinli/metadata_random.csv",
        os.path.join(multinli_dir, "data", "metadata_random.csv"),
        remove=False
    )


def download_waterbirds(data_path):
    logging.info("Downloading Waterbirds...")
    water_birds_dir = os.path.join(data_path, "waterbirds")
    os.makedirs(water_birds_dir, exist_ok=True)
    water_birds_dir_tar = os.path.join(water_birds_dir, "waterbirds.tar.gz")
    download_and_extract(
        "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
        water_birds_dir_tar,
    )


def download_celeba(data_path):
    logging.info("Downloading CelebA...")
    celeba_dir = os.path.join(data_path, "celeba")
    os.makedirs(celeba_dir, exist_ok=True)
    download_and_extract(
        "https://drive.google.com/uc?id=1mb1R6dXfWbvk3DnlWOBO8pDeoBKOcLE6",
        os.path.join(celeba_dir, "img_align_celeba.zip"),
    )
    download_and_extract(
        "https://drive.google.com/uc?id=1acn0-nE4W7Wa17sIkKB0GtfW4Z41CMFB",
        os.path.join(celeba_dir, "list_eval_partition.txt"),
        remove=False
    )
    download_and_extract(
        "https://drive.google.com/uc?id=11um21kRUuaUNoMl59TCe2fb01FNjqNms",
        os.path.join(celeba_dir, "list_attr_celeba.txt"),
        remove=False
    )


def download_imagenetbg(data_path):
    logging.info("Downloading ImageNet Backgrounds Challenge...")
    bg_dir = os.path.join(data_path, "backgrounds_challenge")
    os.makedirs(bg_dir, exist_ok=True)
    download_and_extract(
        "https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz",
        os.path.join(bg_dir, "backgrounds_challenge_data.tar.gz"),
        remove=True
    )
    download_and_extract(
        "https://www.dropbox.com/s/0vv2qsc4ywb4z5v/original.tar.gz?dl=1",
        os.path.join(bg_dir, "original.tar.gz"),
        remove=True
    )
    download_and_extract(
        "https://www.dropbox.com/s/8w29bg9niya19rn/in9l.tar.gz?dl=1",
        os.path.join(bg_dir, "in9l.tar.gz"),
        remove=True
    )


def download_metashift(data_path):
    logging.info("Downloading MetaShift Cats vs. Dogs...")
    ms_dir = os.path.join(data_path, "metashift")
    os.makedirs(ms_dir, exist_ok=True)
    download_and_extract(
        "https://www.dropbox.com/s/a7k65rlj4ownyr2/metashift.tar.gz?dl=1",
        os.path.join(ms_dir, "metashift.tar.gz"),
        remove=True
    )


def download_nicopp(data_path):
    logging.info("Downloading NICO++...")
    bg_dir = os.path.join(data_path, "nicopp")
    os.makedirs(bg_dir, exist_ok=True)
    download_and_extract(
        "https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AACvgYpfQHfS7u_M3yJwkK-ra/track_1/track_1.zip?dl=1",
        os.path.join(bg_dir, "track_1.zip"),
        remove=True
    )
    download_and_extract(
        "https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADMZPkoNJVI4IP1qbLuKtlFa/track_1/dg_label_id_mapping.json?dl=1",
        os.path.join(bg_dir, "dg_label_id_mapping.json"),
        remove=False
    )


def download_breeds(data_path):
    from robustness.tools.breeds_helpers import setup_breeds

    imagenet_dir = Path(data_path)/'imagenet'
    assert (imagenet_dir/'train'/'n06359193'/'n06359193_64665.JPEG').is_file(), \
        'Please download Imagenet (ILSVRC2012) and link it to an "imagenet" folder in your data directory.'
    assert (imagenet_dir/'val'/'n03661043'/'ILSVRC2012_val_00039081.JPEG').is_file(), \
        'Please download Imagenet (ILSVRC2012) and preprocess the validation set appropriately.'

    info_dir = Path(data_path)/'breeds'
    os.makedirs(info_dir, exist_ok=True)
    setup_breeds(info_dir)


def download_cmnist(data_path):
    from torchvision import datasets
    sub_dir = Path(data_path)/'cmnist'
    train_mnist = datasets.mnist.MNIST(sub_dir, train=True, download=True)
    test_mnist = datasets.mnist.MNIST(sub_dir, train=False, download=True)


def generate_metadata(data_path, datasets=['celeba', 'waterbirds', 'civilcomments', 'multinli']):
    dataset_metadata_generators = {
        'celeba': generate_metadata_celeba,
        'waterbirds': generate_metadata_waterbirds,
        'civilcomments': generate_metadata_civilcomments,
        'multinli': generate_metadata_multinli,
        'imagenetbg': generate_metadata_imagenetbg,
        'metashift': generate_metadata_metashift,
        'nico++': generate_metadata_nicopp,
        'mimic_cxr': generate_metadata_mimic_cxr,
        'mimic_notes': generate_metadata_mimic_notes,
        'cxr_multisite': generate_metadata_cxr_multisite,
        'chexpert': generate_metadata_chexpert,
        'breeds': generate_metadata_breeds,
        'cmnist': generate_metadata_cmnist
    }
    for dataset in datasets:
        dataset_metadata_generators[dataset](data_path)


def generate_metadata_celeba(data_path):
    logging.info("Generating metadata for CelebA...")
    with open(os.path.join(data_path, "celeba/list_eval_partition.txt"), "r") as f:
        splits = f.readlines()

    with open(os.path.join(data_path, "celeba/list_attr_celeba.txt"), "r") as f:
        attrs = f.readlines()[2:]

    f = open(os.path.join(data_path, "celeba", "metadata_celeba.csv"), "w")
    f.write("id,filename,split,y,a\n")

    for i, (split, attr) in enumerate(zip(splits, attrs)):
        fi, si = split.strip().split()
        ai = attr.strip().split()[1:]
        yi = 1 if ai[9] == "1" else 0
        gi = 1 if ai[20] == "1" else 0
        f.write("{},{},{},{},{}\n".format(i + 1, fi, si, yi, gi))

    f.close()


def generate_metadata_waterbirds(data_path):
    logging.info("Generating metadata for Waterbirds...")
    df = pd.read_csv(os.path.join(data_path, "waterbirds/waterbird_complete95_forest2water2/metadata.csv"))
    df = df.rename(columns={"img_id": "id", "img_filename": "filename", "place": "a"})

    df[["id", "filename", "split", "y", "a"]].to_csv(
        os.path.join(data_path, "waterbirds", "metadata_waterbirds.csv"), index=False
    )


def generate_metadata_civilcomments(data_path):
    logging.info("Generating metadata for CivilComments...")
    df = pd.read_csv(
        os.path.join(data_path, "civilcomments", "all_data_with_identities.csv"),
        index_col=0,
    )
    group_attrs = [
        "male",
        "female",
        "LGBTQ",
        "christian",
        "muslim",
        "other_religions",
        "black",
        "white",
    ]
    cols_to_keep = ["comment_text", "split", "toxicity"]
    df = df[cols_to_keep + group_attrs]
    df = df.rename(columns={"toxicity": "y"})
    df["y"] = (df["y"] >= 0.5).astype(int)
    df[group_attrs] = (df[group_attrs] >= 0.5).astype(int)
    df["no active attributes"] = 0
    df.loc[(df[group_attrs].sum(axis=1)) == 0, "no active attributes"] = 1

    few_groups, all_groups = [], []
    train_df = df.groupby("split").get_group("train")
    split_df = train_df.rename(columns={"no active attributes": "a"})
    few_groups.append(split_df[["y", "split", "comment_text", "a"]])

    for split, split_df in df.groupby("split"):
        for i, attr in enumerate(group_attrs):
            test_df = split_df.loc[
                split_df[attr] == 1, ["y", "split", "comment_text"]
            ].copy()
            test_df["a"] = i
            all_groups.append(test_df)
            if split != "train":
                few_groups.append(test_df)

    few_groups = pd.concat(few_groups).reset_index(drop=True)
    all_groups = pd.concat(all_groups).reset_index(drop=True)

    for name, df in {"coarse": few_groups, "fine": all_groups}.items():
        df.index.name = "filename"
        df = df.reset_index()
        df["id"] = df["filename"]
        df["split"] = df["split"].replace({"train": 0, "val": 1, "test": 2})
        text = df.pop("comment_text")

        df[["id", "filename", "split", "y", "a"]].to_csv(
            os.path.join(data_path, "civilcomments", f"metadata_civilcomments_{name}.csv"), index=False
        )
        text.to_csv(
            os.path.join(data_path, "civilcomments", f"civilcomments_{name}.csv"),
            index=False,
        )


def generate_metadata_multinli(data_path):
    logging.info("Generating metadata for MultiNLI...")
    df = pd.read_csv(
        os.path.join(data_path, "multinli", "data", "metadata_random.csv"), index_col=0
    )
    df = df.rename(columns={"gold_label": "y", "sentence2_has_negation": "a"})
    df = df.reset_index(drop=True)
    df.index.name = "id"
    df = df.reset_index()
    df["filename"] = df["id"]
    df = df.reset_index()[["id", "filename", "split", "y", "a"]]
    df.to_csv(os.path.join(data_path, "multinli", "metadata_multinli.csv"), index=False)


def generate_metadata_metashift(data_path, test_pct=0.25, val_pct=0.1):
    logging.info("Generating metadata for MetaShift...")
    dirs = {
        'train/cat/cat(indoor)': [1, 1],
        'train/dog/dog(outdoor)': [0, 0],
        'test/cat/cat(outdoor)': [1, 0],
        'test/dog/dog(indoor)': [0, 1]
    }
    ms_dir = os.path.join(data_path, "metashift")

    all_data = []
    for dir in dirs:
        folder_path = os.path.join(ms_dir, 'MetaShift-Cat-Dog-indoor-outdoor', dir)
        y = dirs[dir][0]
        g = dirs[dir][1]
        for img_path in Path(folder_path).glob('*.jpg'):
            all_data.append({
                'filename': img_path,
                'y': y,
                'a': g
            })
    df = pd.DataFrame(all_data)

    rng = np.random.RandomState(42)

    test_idxs = rng.choice(np.arange(len(df)), size=int(len(df) * test_pct), replace=False)
    val_idxs = rng.choice(np.setdiff1d(np.arange(len(df)), test_idxs), size=int(len(df) * val_pct), replace=False)

    split_array = np.zeros((len(df), 1))
    split_array[val_idxs] = 1
    split_array[test_idxs] = 2

    df['split'] = split_array.astype(int)
    df.to_csv(os.path.join(ms_dir, "metadata_metashift.csv"), index=False)


def generate_metadata_imagenetbg(data_path):
    logging.info("Generating metadata for ImagenetBG...")
    bg_dir = Path(os.path.join(data_path, "backgrounds_challenge"))
    dirs = {
        'train': 'in9l/train',
        'val': 'in9l/val',
        'test': 'bg_challenge/original/val',
        'mixed_rand': 'bg_challenge/mixed_rand/val',
        'only_fg': 'bg_challenge/only_fg/val',
        'no_fg': 'bg_challenge/no_fg/val',
    }
    classes = {
        0: 'dog',
        1: 'bird',
        2: 'wheeled vehicle',
        3: 'reptile',
        4: 'carnivore',
        5: 'insect',
        6: 'musical instrument',
        7: 'primate',
        8: 'fish'
    }

    all_data = []
    for dir in dirs:
        for label in classes:
            label_folder = f'0{label}_{classes[label]}'
            folder_path = bg_dir/dirs[dir]/label_folder
            for img_path in folder_path.glob('*.JPEG'):
                all_data.append({
                    'split': dir,
                    'filename': img_path,
                    'y': label,
                    'a': 0
                })

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(bg_dir, "metadata.csv"), index=False)


def generate_metadata_nicopp(data_path):
    logging.info("Generating metadata for NICO++...")
    sub_dir = Path(os.path.join(data_path, "nicopp"))
    attributes = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']   # 6 attributes, 60 labels
    meta = json.load(open(sub_dir/'dg_label_id_mapping.json', 'r'))

    def make_balanced_testset(df, seed=666, verbose=True, num_samples_val_test=75):
        # each group has a test set size of (2/3 * num_samples_val_test) and a val set size of
        # (1/3 * num_samples_val_test); if total samples in original group < num_samples_val_test,
        # val/test will still be split by 1:2, but no training samples remained
        import random
        random.seed(seed)
        val_set, test_set = [], []
        for g in pd.unique(df['g']):
            df_group = df[df['g'] == g]
            curr_data = df_group['filename'].values
            random.shuffle(curr_data)
            split_size = min(len(curr_data), num_samples_val_test)
            val_set += list(curr_data[:split_size // 3])
            test_set += list(curr_data[split_size // 3:split_size])
        if verbose:
            print(f"Val: {len(val_set)}\nTest: {len(test_set)}")
        assert len(set(val_set).intersection(set(test_set))) == 0
        combined_set = dict(zip(val_set, [1 for _ in range(len(val_set))]))
        combined_set.update(dict(zip(test_set, [2 for _ in range(len(test_set))])))
        df['split'] = df['filename'].map(combined_set)
        df['split'].fillna(0, inplace=True)
        df['split'] = df.split.astype(int)
        return df

    all_data = []
    for c, attr in enumerate(attributes):
        for label in meta:
            folder_path = sub_dir/'public_dg_0416'/'train'/attr/label
            y = meta[label]
            for img_path in Path(folder_path).glob('*.jpg'):
                all_data.append({
                    'filename': img_path,
                    'y': y,
                    'a': c
                })
    df = pd.DataFrame(all_data)
    df['g'] = df['a'] + df['y'] * len(attributes)
    df = make_balanced_testset(df)
    df = df.drop(columns=['g'])
    df.to_csv(os.path.join(sub_dir, "metadata.csv"), index=False)


def generate_metadata_mimic_cxr(data_path, test_pct=0.1, val_pct=0.05):
    logging.info("Generating metadata for MIMIC-CXR No Finding Prediction...")
    img_dir = Path(os.path.join(data_path, "MIMIC-CXR-JPG"))

    assert (img_dir/'mimic-cxr-2.0.0-metadata.csv.gz').is_file()
    assert (img_dir/'patients.csv.gz').is_file(), \
        'Please download patients.csv.gz and admissions.csv.gz from MIMIC-IV and place it in the image folder.'
    assert (img_dir/'files/p19/p19316207/s55102753/31ec769b-463d6f30-a56a7e09-76716ec1-91ad34b6.jpg').is_file()

    def ethnicity_mapping(x):
        if pd.isnull(x):
            return 2
        elif x.startswith("WHITE"):
            return 0
        elif x.startswith("BLACK"):
            return 1
        return 2

    patients = pd.read_csv(img_dir/'patients.csv.gz')
    ethnicities = pd.read_csv(img_dir/'admissions.csv.gz').drop_duplicates(
        subset=['subject_id']).set_index('subject_id')['race'].to_dict()
    patients['ethnicity'] = patients['subject_id'].map(ethnicities).map(ethnicity_mapping)
    labels = pd.read_csv(img_dir/'mimic-cxr-2.0.0-negbio.csv.gz')
    meta = pd.read_csv(img_dir/'mimic-cxr-2.0.0-metadata.csv.gz')

    df = meta.merge(patients, on='subject_id').merge(labels, on=['subject_id', 'study_id'])
    df['age_decile'] = pd.cut(df['anchor_age'], bins=list(range(0, 101, 10))).apply(lambda x: f'{x.left}-{x.right}').astype(str)
    df['frontal'] = df.ViewPosition.isin(['AP', 'PA'])

    df['filename'] = df.apply(
        lambda x: os.path.join(
            img_dir,
            'files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'
        ), axis=1)
    df = df[df.anchor_age > 0]

    attr_mapping = {'M_0': 0, 'F_0': 1, 'M_1': 2, 'F_1': 3, 'M_2': 4, 'F_2': 5}
    df['a'] = (df['gender'] + '_' + df['ethnicity'].astype(str)).map(attr_mapping)
    df['y'] = df['No Finding'].fillna(0.0).astype(int)

    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42, stratify=df['a'])
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42, stratify=df.loc[train_val_idx, 'a'])

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    (img_dir/'subpop_bench_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(img_dir, 'subpop_bench_meta', "metadata_no_finding.csv"), index=False)


def generate_metadata_mimic_notes(data_path):
    logging.info("Generating metadata for MIMIC-III mortality prediction...")
    sub_dir = Path(os.path.join(data_path, "mimic_notes"))

    assert (sub_dir/'cohort.pkl').is_file()
    assert (sub_dir/'features.npy').is_file()

    df = pd.read_pickle(sub_dir/'cohort.pkl')
    df['y'] = df['mort_icu']
    df['a'] = (df['gender'] == 'M')
    df['filename'] = df['array_index']
    df['split'] = 0
    df.loc[df.fold_id == 'eval', 'split'] = 1
    df.loc[df.fold_id == 'test', 'split'] = 2

    (sub_dir/'subpop_bench_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(sub_dir, 'subpop_bench_meta', "metadata.csv"), index=False)


def generate_metadata_cxr_multisite(data_path, test_pct=0.1, val_pct=0.05):
    logging.info("Generating metadata for MIMIC-CXR and CheXpert pnuemonia prediction...")
    mimic_dir = Path(os.path.join(data_path, "MIMIC-CXR-JPG"))
    chexpert_dir = Path(os.path.join(data_path, "chexpert"))

    assert (mimic_dir/'mimic-cxr-2.0.0-metadata.csv.gz').is_file()
    assert (mimic_dir/'files/p19/p19316207/s55102753/31ec769b-463d6f30-a56a7e09-76716ec1-91ad34b6.jpg').is_file()
    assert (chexpert_dir/'train.csv').is_file()
    assert (chexpert_dir/'train/patient48822/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'valid/patient64636/study1/view1_frontal.jpg').is_file()

    labels_mimic = pd.read_csv(mimic_dir/'mimic-cxr-2.0.0-negbio.csv.gz')
    meta_mimic = pd.read_csv(mimic_dir/'mimic-cxr-2.0.0-metadata.csv.gz')

    df_mimic = meta_mimic.merge(labels_mimic, on=['subject_id', 'study_id'])

    df_mimic['filename'] = df_mimic.apply(
        lambda x: os.path.join(
            mimic_dir,
            'files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'
        ), axis=1)

    df_mimic['site'] = 'MIMIC-CXR'
    df_mimic['a'] = 0

    df_cxp = pd.concat([pd.read_csv(chexpert_dir/'train.csv'), pd.read_csv(chexpert_dir/'valid.csv')],
                       ignore_index=True)

    df_cxp['filename'] = df_cxp['Path'].astype(str).apply(lambda x: os.path.join(chexpert_dir, x[x.index('/')+1:]))
    df_cxp['site'] = 'CheXpert'
    df_cxp['a'] = 1

    cols_to_take = ['filename', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding',
                    'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'site', 'a']

    df = pd.concat((df_mimic[cols_to_take], df_cxp[cols_to_take]), ignore_index=True)
    df['y'] = df['Pneumonia'].fillna(0).astype(int).map({-1: 0, 1: 1, 0: 0})

    # 90% of healthy patients are from MIMIC-CXR
    healthy = df[df.y == 0]
    n_to_keep = int((healthy.a == 0).sum()/0.9) - (healthy.a == 0).sum()
    healthy_90 = pd.concat((healthy[healthy.a == 0], healthy[healthy.a == 1].sample(
        n=n_to_keep, random_state=42, replace=False)), ignore_index=True)

    # 90% of sick patients are from CheXpert
    sick = df[df.y == 1]
    n_to_keep = int(sick.a.sum()/0.9) - sick.a.sum()
    sick_90 = pd.concat((sick[sick.a == 1], sick[sick.a == 0].sample(
        n=n_to_keep, random_state=42, replace=False)), ignore_index=True)

    df_final = pd.concat((healthy_90, sick_90), ignore_index=True)
    train_val_idx, test_idx = train_test_split(df_final.index, test_size=test_pct, random_state=42, stratify=df_final['y'])
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42, stratify=df_final.loc[train_val_idx, 'y'])

    df_final['split'] = 0
    df_final.loc[val_idx, 'split'] = 1
    df_final.loc[test_idx, 'split'] = 2

    # construct additional dataset with opposite correlation
    # remove train and val set samples
    healthy = df[(df.y == 0) & (~healthy.filename.isin(df_final[df_final.split.isin([0, 1])].filename))]
    n_to_keep = int(healthy.a.sum() / 0.9) - healthy.a.sum()
    healthy_10 = pd.concat((healthy[healthy.a == 1], healthy[healthy.a == 0].sample(
        n=n_to_keep, random_state=42, replace=False)), ignore_index=True)
    sick = df[(df.y == 1) & (~sick.filename.isin(df_final[df_final.split.isin([0, 1])].filename))]
    n_to_keep = int((sick.a == 0).sum()/0.9) - (sick.a == 0).sum()
    # not enough sick patients in CheXpert
    sick_10 = pd.concat((sick[sick.a == 0], sick[sick.a == 1].sample(
        n=n_to_keep, random_state=42, replace=True)), ignore_index=True)
    # subsample from ~250k to save time
    df_eval = pd.concat((healthy_10, sick_10), ignore_index=True).sample(n=1024*16, random_state=42, replace=False)
    df_eval['split'] = 3

    df = pd.concat((df_final, df_eval), ignore_index=True)
    (mimic_dir/'subpop_bench_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(mimic_dir, 'subpop_bench_meta', "metadata_multisite.csv"), index=False)


def generate_metadata_chexpert(data_path, test_pct=0.15, val_pct=0.1):
    logging.info("Generating metadata for CheXpert No Finding prediction...")
    chexpert_dir = Path(os.path.join(data_path, "chexpert"))
    assert (chexpert_dir/'train.csv').is_file()
    assert (chexpert_dir/'train/patient48822/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'valid/patient64636/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'CHEXPERT DEMO.xlsx').is_file()

    df = pd.concat([pd.read_csv(chexpert_dir/'train.csv'), pd.read_csv(chexpert_dir/'valid.csv')], ignore_index=True)

    df['filename'] = df['Path'].astype(str).apply(lambda x: os.path.join(chexpert_dir, x[x.index('/')+1:]))
    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:])).astype(str)
    df = df[df.Sex.isin(['Male', 'Female'])]
    details = pd.read_excel(chexpert_dir/'CHEXPERT DEMO.xlsx', engine='openpyxl')[['PATIENT', 'PRIMARY_RACE']]
    details['subject_id'] = details['PATIENT'].apply(lambda x: x[7:]).astype(int).astype(str)

    df = pd.merge(df, details, on='subject_id', how='inner').reset_index(drop=True)

    def cat_race(r):
        if isinstance(r, str):
            if r.startswith('White'):
                return 0
            elif r.startswith('Black'):
                return 1
        return 2

    df['ethnicity'] = df['PRIMARY_RACE'].apply(cat_race)
    attr_mapping = {'Male_0': 0, 'Female_0': 1, 'Male_1': 2, 'Female_1': 3, 'Male_2': 4, 'Female_2': 5}
    df['a'] = (df['Sex'] + '_' + df['ethnicity'].astype(str)).map(attr_mapping)
    df['y'] = df['No Finding'].fillna(0.0).astype(int)

    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42, stratify=df['a'])
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42, stratify=df.loc[train_val_idx, 'a'])

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    (chexpert_dir/'subpop_bench_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(chexpert_dir, 'subpop_bench_meta', "metadata_no_finding.csv"), index=False)


def generate_metadata_breeds(data_path, val_pct=0.1):
    from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26, print_dataset_info, ClassHierarchy
    from robustness.tools import folder
    from robustness.tools.helpers import get_label_mapping

    info_dir = Path(data_path)/'breeds'
    imagenet_dir = Path(data_path)/'imagenet'
    assert (imagenet_dir/'train'/'n06359193'/'n06359193_64665.JPEG').is_file(), \
        'Please download Imagenet (ILSVRC2012) and link it to an "imagenet" folder in your data directory'
    assert (imagenet_dir/'val'/'n03661043'/'ILSVRC2012_val_00039081.JPEG').is_file(), \
        'Please download Imagenet (ILSVRC2012) and preprocess the validation set appropriately.'

    dss = {
        'living17': make_living17,
        'entity13': make_entity13,
        'entity30': make_entity30,
        'nonliving26': make_nonliving26
    }
    hier = ClassHierarchy(info_dir)

    for ds, fn in dss.items():
        logging.info("-"*50)
        logging.info(f"Generating metadata for {ds}...")
        superclasses, subclass_split, label_map = fn(info_dir, split="rand")
        source_subclasses, target_subclasses = subclass_split

        print(print_dataset_info(superclasses, subclass_split, label_map, hier.LEAF_NUM_TO_NAME))

        label_mapping_source = get_label_mapping('custom_imagenet', source_subclasses)
        label_mapping_target = get_label_mapping('custom_imagenet', target_subclasses)

        source_train_set = folder.ImageFolder(
            root=imagenet_dir/'train', transform=None, label_mapping=label_mapping_source)
        source_test_set = folder.ImageFolder(
            root=imagenet_dir/'val', transform=None, label_mapping=label_mapping_source)
        target_set = folder.ImageFolder(
            root=imagenet_dir/'val', transform=None, label_mapping=label_mapping_target)

        # train
        df = pd.DataFrame(source_train_set.imgs, columns=['filename', 'y']).assign(split=0).reset_index(drop=True)
        _, val_idxs = train_test_split(df.index, test_size=val_pct, random_state=42, stratify=df['y'])
        # val
        df.loc[val_idxs, 'split'] = 1
        # test
        df = pd.concat((df, pd.DataFrame(
            source_test_set.imgs, columns=['filename', 'y']).assign(split=2))).reset_index(drop=True)
        # zero-shot
        df = pd.concat((df, pd.DataFrame(
            target_set.imgs, columns=['filename', 'y']).assign(split=3))).reset_index(drop=True)
        df['a'] = 0
        df.to_csv(os.path.join(info_dir, f"metadata_{ds}.csv"), index=False)


def generate_metadata_cmnist(data_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset')
    parser.add_argument('datasets', nargs='+', type=str, default=[
        'celeba', 'waterbirds', 'civilcomments', 'multinli', 'imagenetbg', 'metashift', 'nico++',
        'mimic_cxr', 'chexpert', 'mimic_notes', 'cxr_multisite', 'breeds', 'cmnist'])
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--download', action='store_true', default=False)
    args = parser.parse_args()

    if args.download:
        download_datasets(args.data_path, args.datasets)
    generate_metadata(args.data_path, args.datasets)
