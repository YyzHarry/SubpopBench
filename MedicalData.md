# Downloading Clinical Data

The `MIMICNoFinding`, `CheXpertNoFinding`, `CXRMultisite`, and `MIMICNotes` datasets are not able to be downloaded through `scripts/download.py` as they require additional steps to gain access.

## MIMICNoFinding

1. [Obtain access](https://mimic-cxr.mit.edu/about/access/) to the MIMIC-CXR-JPG Database on PhysioNet and download the [dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). We recommend downloading from the GCP bucket:

```bash
gcloud auth login
mkdir MIMIC-CXR-JPG
gsutil -m rsync -d -r gs://mimic-cxr-jpg-2.0.0.physionet.org MIMIC-CXR-JPG
```

2. In order to obtain demographic information for each patient, you will need to obtain access to [MIMIC-IV](https://physionet.org/content/mimiciv/). Download `core/patients.csv.gz` and `core/admissions.csv.gz` and place the files in the `MIMIC-CXR-JPG` directory.

3. Move or create a symbolic link to the `MIMIC-CXR-JPG` folder from your data directory.

4. Run `python -m subpopbench.scripts.download mimic_cxr --data_path <data_path>`.

5. (Optional) As the original jpgs have very high resolution, caching the images as downsampled copies might speed things up if you are training a lot of models. In this case, you should run `python -m subpopbench.scripts.cache_mimic_cxr --data_path <data_path>`.

## CheXpertNoFinding

1. Download the [downsampled CheXpert dataset](http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip) and extract it.

2. Register for an account and download the CheXpert demographics data [here](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf). Place the `CHEXPERT DEMO.xlsx` in your CheXpert directory. 

3. Move or create a symbolic link to the `CheXpert-v1.0-small` folder named `chexpert` in your data directory.

4. Run `python -m subpopbench.scripts.download chexpert --data_path <data_path>`.

## CXRMultisite

1. Acquire both of the datasets above.

2. Run `python -m subpopbench.scripts.download cxr_multisite --data_path <data_path>`.

## MIMICNotes

1. [Obtain access](https://mimic.mit.edu/docs/gettingstarted/) to the [MIMIC-III Database](https://physionet.org/content/mimiciii/) on PhysioNet. 

2. Follow the [instructions here](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/buildmimic/postgres/README.md) to load MIMIC-III into a PostgreSQL database.

3. Modify `scripts/preprocess_mimic_notes.py` to update `output_dir` with your data directory, as well as the database access credentials. Then, run `python -m subpopbench.scripts.preprocess_mimic_notes`.

4. Run `python -m subpopbench.scripts.download mimic_notes --data_path <data_path>`.
