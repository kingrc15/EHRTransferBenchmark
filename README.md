# eICU Preprocessing

## Citation

This repository is a modification of the data preprocessing code used in **[Benchmarking machine learning models on multi-centre eICU critical care dataset](https://arxiv.org/abs/1910.00964v3)** by Seyedmostafa Sheikhalishahi, Vevake Balaraman and Venet Osmani.

The original repositiory - [eICU_Benchmark_updated](https://github.com/mostafaalishahi/eICU_Benchmark_updated)

Dataset being preprocessed - [eICU paper](https://www.nature.com/articles/sdata2018178) by Tom J. Pollard et. al.

The modification of the eICU_Benchmark_updated code is to adhere to the exclusion criteria and features used in the paper: **[Multitask learning and benchmarking with clinical time series data](https://arxiv.org/abs/1703.07771)** by Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and Aram Galstyan. This paper performs clinical predictions on the MIMIC-III dataset ([MIMIC-III paper](http://www.nature.com/articles/sdata201635)).

Hence, the term **eligible patients** refer to patients in the eICU dataset who adhere to the exclusion criteria in Harutyunyan et al.

## Structure

There are only 2 scripts to run:

1. **data_extraction_root.py** - extracting timeseries data for each eligible patient (the data to be trained on)
2. **create_tasks.py** - creating listfiles which store the ground truths for clinical tasks (the labels)

## Data extraction

Ensure the eICU dataset CSVs are available on the disk.

1. Clone the repository.

2. The following command will:

- Create a directory for each eligible patient stay
- Writes patients demographics into pats.csv
- Nursecharting info into nc.csv
- Lab measurments into lab.csv
- Nurseassessment into na.csv
- Merges these four csv files into one timeseries.csv for each patient stay
- Create a `root/` directory iff a task_dir is specified.
- Truncate data only to a patient's ICU stay, rename files to unique episode number, and throw out patients with under 15 records iff a task_dir is specified.
- Creates timeseries_info.csv file which stores information to determine eligibility for each task iff a task_dir is specified.

> python data_extraction_root.py --eicu_dir "directory of csv files" --output_dir "directory to save the extracted data" --task_dir "directory to save modified data and task listfiles"

## Creating labels

3. The following command will:

- Create train, test, val listfiles for each specified task. Listfiles typically contain 3 components: 1. stay - the patient timeseries file name 2. period_length - amount of time model is allowed to look at (except IHM which is always 48 hrs) 3. y_true - the ground truth (except for pheno as it has multple labels)
- Create regional splits for each task specified

> python create_labels.py --eicu_dir "directory of csv files" --task_dir "directory to save modified data and task listfiles" \[--regional\] \[--all | --ihm | --los | --decomp | --pheno \] \[--skip_labeling\]

Use `--all` for all tasks, `--regional` to perform regional splits.
If you want to only do a specific subset of tasks, use the corresponding flags.
If the root directory has already been fully populated, use `--skip_labeling`.

### Linking to root

4. For training, validation, and testing you need to have the data in the same folder as the task folder (train and validation data in `train/` and test data in `test/`).

This can be accomplished by symbolically linking a `train` and `test` folder to the `root/` directory.

For linux, fill the `ABSOLUTE_PATH_TO_SPLITS` variable in `symlink.sh` then run:

```bash
chmod +x symlink.sh
./symlink.sh
```
