import os
import argparse
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


class CXRDataset(Dataset):

    def __init__(self, df, min_dim, overwrite=False):
        super().__init__()
        self.df = df
        self.overwrite = overwrite
        self.transform = transforms.Resize(min_dim)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = df.iloc[idx]
        img_path = Path(row['filename'])
        reduced_img_path = list(img_path.parts)
        assert reduced_img_path[-5] == 'files', reduced_img_path
        reduced_img_path[-5] = 'downsampled_files'
        reduced_img_path = Path(*reduced_img_path).with_suffix('.png')

        if self.overwrite or not reduced_img_path.is_file():
            reduced_img_path.parent.mkdir(exist_ok=True, parents=True)
            img = Image.open(img_path).convert("RGB")
            new_img = self.transform(img)
            new_img.save(reduced_img_path)
            return 1
        else:
            return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample MIMIC-CXR')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--n_workers', type=int, default=64)
    parser.add_argument('--min_dim', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing cached files')
    args = parser.parse_args()

    mimic_dir = Path(os.path.join(args.data_path, "MIMIC-CXR-JPG"))

    assert (mimic_dir/'files/p19/p19316207/s55102753/31ec769b-463d6f30-a56a7e09-76716ec1-91ad34b6.jpg').is_file()
    assert (mimic_dir/'subpop_bench_meta'/'metadata_no_finding.csv').is_file(), \
        "Please run download.py to generate the metadata for `mimic_cxr` first!"

    df = pd.read_csv(mimic_dir/'subpop_bench_meta'/'metadata_no_finding.csv')
    ds = CXRDataset(df, args.min_dim)
    dl = DataLoader(ds, batch_size=64, num_workers=args.n_workers, shuffle=False)

    for i in tqdm(dl):
        pass
