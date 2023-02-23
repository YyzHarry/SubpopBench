# Adapted from https://github.com/irenetrampoline/mimic-disparities/tree/master

import psycopg2
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


def save(df, features, vocab, meta, path):
    df.to_pickle(path/'cohort.pkl')
    np.save(path/'features.npy', features)
    pickle.dump(vocab, (path/'vocab.pkl').open('wb'))
    pickle.dump(meta, (path/'meta.pkl').open('wb'))


def split(df, eval_weight=1, target_name='target'):
    n_splits_train = 5
    idx_train = df[df.fold_id == 'train'].index
    skf = StratifiedKFold(n_splits=n_splits_train + eval_weight, random_state=42, shuffle=True)
    for c, (_, fold_index) in enumerate(skf.split(idx_train, df.loc[idx_train, target_name])):
        df.loc[idx_train[fold_index], 'fold_id'] = str(c)
    df.loc[df.fold_id.isin(list(map(str, range(n_splits_train, n_splits_train + eval_weight, 1)))), 'fold_id'] = 'eval'
    return df


def bin_age(age):
    bins = [
        (18, 30),
        (30, 45),
        (45, 55),
        (55, 65),
        (65, 75),
        (75, 90),
    ]

    for c, i in enumerate(bins):
        if (i[0] <= age < i[1]) or (c == len(bins) - 1 and age == i[1]):
            return f'{i[0]}-{i[1]}'


# TODO: change to your own path
output_dir = Path('/path/to/your/subpop_bench/output/dir')
output_dir.mkdir(parents=True, exist_ok=True)

# TODO: update your own database credentials
sqluser = 'username'
dbname = 'mimic'
schema_name = 'mimiciii'

random.seed(22891)

# Connect to local postgres version of mimic
con = psycopg2.connect(dbname=dbname, user=sqluser, host=dbname, password='password')
cur = con.cursor()
cur.execute('SET search_path to ' + schema_name)


# ======== helper function for imputing missing values

def replace(group):
    """
    takes in a pandas group, and replaces the null value with the mean of the none null values of the same group
    """
    mask = group.isnull()
    group[mask] = group[~mask].mean()
    return group


# ======== get the icu details

# this query extracts the following:
#   Unique ids for the admission, patient and icu stay
#   Patient gender 
#   admission & discharge times 
#   length of stay 
#   age 
#   ethnicity 
#   admission type 
#   in hospital death?
#   in icu death?
#   one year from admission death?
#   first hospital stay 
#   icu intime, icu outime 
#   los in icu 
#   first icu stay?

denquery = \
"""
-- This query extracts useful demographic/administrative information for patient ICU stays
--DROP MATERIALIZED VIEW IF EXISTS icustay_detail CASCADE;
--CREATE MATERIALIZED VIEW icustay_detail as
--ie is the icustays table 
--adm is the admissions table 
SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
, pat.gender
, adm.admittime, adm.dischtime, adm.diagnosis
, ROUND( (CAST(adm.dischtime AS DATE) - CAST(adm.admittime AS DATE)) , 4) AS los_hospital
, ROUND( (CAST(adm.admittime AS DATE) - CAST(pat.dob AS DATE))  / 365, 4) AS age
, adm.ethnicity, adm.ADMISSION_TYPE
--, adm.hospital_expire_flag
, adm.insurance
, CASE when adm.deathtime between adm.admittime and adm.dischtime THEN 1 ELSE 0 END AS mort_hosp
, CASE when adm.deathtime between ie.intime and ie.outtime THEN 1 ELSE 0 END AS mort_icu
, CASE when adm.deathtime between adm.admittime and adm.admittime + interval '365' day  THEN 1 ELSE 0 END AS mort_oneyr
, DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN 1
    ELSE 0 END AS first_hosp_stay
-- icu level factors
, ie.intime, ie.outtime
, ie.FIRST_CAREUNIT
, ROUND( (CAST(ie.outtime AS DATE) - CAST(ie.intime AS DATE)) , 4) AS los_icu
, DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq
-- first ICU stay *for the current hospitalization*
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN 1
    ELSE 0 END AS first_icu_stay
FROM icustays ie
INNER JOIN admissions adm
    ON ie.hadm_id = adm.hadm_id
INNER JOIN patients pat
    ON ie.subject_id = pat.subject_id
WHERE adm.has_chartevents_data = 1
ORDER BY ie.subject_id, adm.admittime, ie.intime;
"""

den = pd.read_sql_query(denquery, con)

# ---- drop patients with less than 48 hour
den['los_icu_hr'] = (den.outtime - den.intime).astype('timedelta64[h]')
den = den[(den.los_icu_hr >= 48)]
den = den[(den.age < 300)]
den.drop('los_icu_hr', 1, inplace=True)

# ---- clean up

# micu --> medical 
# csru --> cardiac surgery recovery unit 
# sicu --> surgical icu 
# tsicu --> Trauma Surgical Intensive Care Unit
# NICU --> Neonatal 

den['adult_icu'] = np.where(den['first_careunit'].isin(['PICU', 'NICU']), 0, 1)

# no need to yell 
den.ethnicity = den.ethnicity.str.lower()
den.ethnicity.loc[(den.ethnicity.str.contains('^white'))] = 'white'
den.ethnicity.loc[(den.ethnicity.str.contains('^black'))] = 'black'
den.ethnicity.loc[~(den.ethnicity.str.contains('|'.join(['white', 'black'])))] = 'other'

# den = pd.concat([den, pd.get_dummies(den['ethnicity'], prefix='eth')], 1)
# den = pd.concat([den, pd.get_dummies(den['admission_type'], prefix='admType')], 1)

den.drop(['diagnosis', 'hospstay_seq', 'los_icu', 'icustay_seq', 'admittime', 'dischtime', 'los_hospital', 'intime',
          'outtime', 'first_careunit'], 1, inplace=True)

den = den[(den['adult_icu'] == 1)].dropna()

notesquery = \
"""
SELECT fin.subject_id, fin.hadm_id, fin.icustay_id, string_agg(fin.text, ' ') as chartext
FROM (
  select ie.subject_id, ie.hadm_id, ie.icustay_id, ne.text
  from icustays ie
  left join noteevents ne
  on ie.subject_id = ne.subject_id and ie.hadm_id = ne.hadm_id 
  and ne.charttime between ie.intime and ie.intime + interval '48' hour
  --and ne.iserror != '1'
  where ne.category != 'Discharge summary'
) fin 
group by fin.subject_id, fin.hadm_id, fin.icustay_id 
order by fin.subject_id, fin.hadm_id, fin.icustay_id; 
"""

notes48 = pd.read_sql_query(notesquery, con)

output_df = notes48.merge(den, on=['subject_id', 'hadm_id', 'icustay_id'], how='inner')

print('notes48:', len(notes48))
print('demographics:', len(den))
print('merged:', len(output_df))

output_df['age_group'] = output_df.age.apply(bin_age)

train_val_inds, test_inds = train_test_split(output_df.index, test_size=0.25, random_state=42)
output_df['fold_id'] = None
output_df.loc[train_val_inds, 'fold_id'] = 'train'
output_df.loc[test_inds, 'fold_id'] = 'test'
output_df = split(output_df, target_name='mort_hosp')

n_tfidf_features = 10000
tfidf = TfidfVectorizer(max_features=n_tfidf_features).fit(
    output_df.query('fold_id in ["0", "1", "2", "3", "4"]').chartext.values)
features = tfidf.transform(output_df.chartext.values)

output_df = output_df.reset_index().rename(columns={'index': 'array_index'})

vocab = tfidf.get_feature_names()

meta = {
        'targets': ['mort_hosp'],
        'groups': ['gender', 'ethnicity', 'age_group'],
        'vocab': vocab
    }

(output_dir/'mimic_notes').mkdir(exist_ok=True)
save(output_df, features, vocab, meta, output_dir/'mimic_notes')
