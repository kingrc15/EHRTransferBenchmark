import pandas as pd
import os
import numpy as np
import gc


def get_phenotype_order():
    label_pheno_global_order = [ "Acute and unspecified renal failure",  "Acute cerebrovascular disease", 
        "Acute myocardial infarction", "Cardiac dysrhythmias", "CKD", "COPD", "Complications of surgical", 
        "Conduction disorders", "CHF", "Coronary athe", "DM with complications", "DM without complication", 
        "lipid disorder", "Essential hypertension", "Fluid disorders", "Gastrointestinal hem", 
        "Hypertension with complications", "Other liver diseases", "lower respiratory", "upper respiratory", 
        "Pleurisy", "Pneumonia", "Respiratory failure", "Septicemia", "Shock",
    ]
    return label_pheno_global_order

# This function from the utils.py file developed by Sheikhalishahi et al.
# Repo: https://github.com/mostafaalishahi/eICU_Benchmark_updated
def dataframe_from_csv(path, header=0, index_col=False):
    return pd.read_csv(path, header=header, index_col=index_col)

# This function from the utils.py file developed by Sheikhalishahi et al.
# Repo: https://github.com/mostafaalishahi/eICU_Benchmark_updated
# Was modified slightly 
def read_diagnosis_table(eicu_path):
    diag = dataframe_from_csv(os.path.join(eicu_path, 'diagnosis.csv.gz'), index_col=False)
    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv.gz'), index_col=False)

    #Clip diagnosis between admission and discharge
    diag = diag[diag["diagnosisoffset"] > 0]
    diag = diag.merge(pats[['patientunitstayid','unitdischargeoffset']], on='patientunitstayid')
    msk = diag["diagnosisoffset"].astype(float) <= diag['unitdischargeoffset'].astype(float)
    diag = diag[msk]

    diag = diag[['patientunitstayid', 'activeupondischarge', 'diagnosisoffset',
                'diagnosisstring', 'icd9code']]
    diag = diag[diag['icd9code'].notnull()]
    tes = diag['icd9code'].str.split(pat=",", expand=True, n=-1)

    labels_name = ["Shock","Septicemia","Respiratory failure","Pneumonia","Pleurisy",
              "upper respiratory","lower respiratory","Other liver diseases",
              "Hypertension with complications","Gastrointestinal hem",
              "Fluid disorders","Essential hypertension","lipid disorder",
              "DM without complication","DM with complications",
              "Coronary athe","CHF", "Conduction disorders","Complications of surgical",
              "COPD", "CKD", "Cardiac dysrhythmias","Acute myocardial infarction",
               "Acute cerebrovascular disease","Acute and unspecified renal failure"]
    
    for i in range(tes.shape[1]):
        diag[f'icd{i}'] = tes[i]
        diag[f'icd{i}'] = diag[f'icd{i}'].str.replace('.', '').astype(str).apply(lambda x: x.strip())
    diag = diag.reindex(columns=diag.columns.tolist() + labels_name)
    diag[labels_name] = np.nan
    return diag, tes.shape[1]

# This function from the utils.py file developed by Sheikhalishahi et al.
# Repo: https://github.com/mostafaalishahi/eICU_Benchmark_updated
# Was modified slightly 
def diag_labels(diag, num):
    import json
    codes = json.load(open('resources/phen_code.json'))
    for i in range(num):
        diag.loc[diag[f'icd{i}'].isin(codes['septicemia']), 'Septicemia'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['Shock']), 'Shock'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['Compl_surgical']), 'Complications of surgical'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['ckd']), 'CKD'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['renal_failure']), 'Acute and unspecified renal failure'] = 1

        diag.loc[diag[f'icd{i}'].isin(codes['Gastroint_hemorrhage']), 'Gastrointestinal hem'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['Other_liver_dis']), 'Other liver diseases'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['upper_respiratory']), 'upper respiratory'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['lower_respiratory']), 'lower respiratory'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['Resp_failure']), 'Respiratory failure'] = 1

        diag.loc[diag[f'icd{i}'].isin(codes['Pleurisy']), 'Pleurisy'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['COPD']), 'COPD'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['Pneumonia']), 'Pneumonia'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['Acute_cerebrovascular']), 'Acute cerebrovascular disease'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['Congestive_hf']), 'CHF'] = 1

        diag.loc[diag[f'icd{i}'].isin(codes['Cardiac_dysr']), 'Cardiac dysrhythmias'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['Conduction_dis']), 'Conduction disorders'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['Coronary_ath']), 'Coronary athe'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['myocar_infarction']), 'Acute myocardial infarction'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['hypercomp']), 'Hypertension with complications'] = 1

        diag.loc[diag[f'icd{i}'].isin(codes['essehyper']), 'Essential hypertension'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['fluiddiso']), 'Fluid disorders'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['lipidmetab']), 'lipid disorder'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['t2dmcomp']), 'DM with complications'] = 1
        diag.loc[diag[f'icd{i}'].isin(codes['t2dmwocomp']), 'DM without complication'] = 1
    return diag


# This is code from the utils.py file developed by Sheikhalishahi et al.
# Repo: https://github.com/mostafaalishahi/eICU_Benchmark_updated
# Was modified slightly 
def get_pheno_listfile(path_to_eicu):
    label_pheno = get_phenotype_order()

    diag_columns = ["patientunitstayid", "itemoffset", "Respiratory failure", "Essential hypertension",
        "Cardiac dysrhythmias", "Fluid disorders", "Septicemia", "Acute and unspecified renal failure",
        "Pneumonia", "Acute cerebrovascular disease", "CHF", "CKD", "COPD", "Acute myocardial infarction",
        "Gastrointestinal hem", "Shock", "lipid disorder", "DM with complications", "Coronary athe",
        "Pleurisy", "Other liver diseases", "lower respiratory", "Hypertension with complications",
        "Conduction disorders", "Complications of surgical", "upper respiratory", "DM without complication",
    ]

    diag, num = read_diagnosis_table(path_to_eicu)
    pats = pd.read_csv(os.path.join(path_to_eicu, 'patient.csv.gz'))
    diag = diag_labels(diag, num)
    diag.dropna(how="all", subset=label_pheno, inplace=True)

    stay_diag = set(diag["patientunitstayid"].unique())
    stay_all = set(pats["patientunitstayid"].unique())
    stay_intersection = stay_all.intersection(stay_diag)

    stay_pheno = list(stay_intersection)

    diag = diag[diag["patientunitstayid"].isin(stay_pheno)]
    diag.rename(index=str, columns={"diagnosisoffset": "itemoffset"}, inplace=True)
    diag = diag[diag_columns]
    label = diag.groupby("patientunitstayid").sum()
    
    pats = pats.set_index("patientunitstayid")
    label = label.dropna()
    label = label.reset_index()
    label = label.rename(columns={"patientunitstayid": "stay", "itemoffset": "period_length"})

    label[label_pheno] = np.where(label[label_pheno] >= 1, 1, label[label_pheno])
    label[label_pheno] = label[label_pheno].astype(int)
    return label[['stay']+label_pheno]

#Developed by Ethan Veselka
def split(name, eicu_path, write=False):
    len_tr = 0
    len_val = 0
    len_test = 0
    non_split = name.split("_")[0]+"/"
    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv.gz'), index_col=False)
    train_list = dataframe_from_csv(os.path.join(non_split, 'train_listfile.csv'), index_col=False)
    test_list = dataframe_from_csv(os.path.join(non_split, 'test_listfile.csv'), index_col=False)
    val_list = dataframe_from_csv(os.path.join(non_split, 'val_listfile.csv'), index_col=False)
    hospitals = dataframe_from_csv(os.path.join(eicu_path, 'hospital.csv'), index_col=False)
    mapper_df = pd.read_csv('resources/episode_mapper.csv')
    mapper = dict(zip(mapper_df['episode'], mapper_df['unitstayid']))
    del mapper_df

    comp_table = pd.concat([train_list, test_list, val_list])
    comp_table["patientunitstayid"] = comp_table["stay"].apply(
        lambda x: mapper[x]
    )

    tr = train_list["stay"].apply(
        lambda x:  mapper[x]
        ).tolist()
    test = test_list["stay"].apply(
        lambda x:  mapper[x]
        ).tolist()
    val = val_list["stay"].apply(
        lambda x:  mapper[x]
        ).tolist()
    print("Train: ", len(tr))
    print("Val: ", len(val))
    print("Test: ", len(test))
    print("\n")
    print("Total: ", len(tr) + len(val) + len(test))
    print("------------------")

    tr_pats = pats[pats["patientunitstayid"].isin(tr)]
    test_pats = pats[pats["patientunitstayid"].isin(test)]
    val_pats = pats[pats["patientunitstayid"].isin(val)]

    merged_tr = pd.merge(tr_pats, hospitals, on="hospitalid")
    merged_test = pd.merge(test_pats, hospitals, on="hospitalid")
    merged_val = pd.merge(val_pats, hospitals, on="hospitalid")

    print("Train:")
    size = merged_tr.groupby("region").size()
    print(size)
    print("\n")

    print("Validation:")
    size = merged_val.groupby("region").size()
    print(size)
    print("\n")

    print("Test:")
    size = merged_test.groupby("region").size()
    print(size)

    if write:

        #############################################################
        # Create and save train listfiles
        patients_visited = set()
        for region, region_group in merged_tr.groupby("region"):
            region = region.lower()
            table = comp_table[
                comp_table["patientunitstayid"].isin(region_group["patientunitstayid"])
            ]
            region_table = table[
                [col for col in table.columns if col != "patientunitstayid"]
            ]


            #ENSURE NO PATIENT IS IN ANY OTHER REGION -- CONRAD
            pid = region_table['stay'].apply(lambda x: x.split("_")[0])
            if pid.isin(patients_visited).astype(int).sum() > 0: 
                print("TRIGGERED", pid[pid.isin(patients_visited)])
                region_table = region_table[~pid.isin(patients_visited)]
            patients_visited = patients_visited | set(pid)


            region_table.to_csv(f"{name}{region}_train.csv", index=False)
            len_tr += len(region_table)

        #############################################################
        # Create and save validation listfiles
        patients_visited = set()
        for region, region_group in merged_val.groupby("region"):
            region = region.lower()
            table = comp_table[
                comp_table["patientunitstayid"].isin(region_group["patientunitstayid"])
            ]
            region_table = table[
                [col for col in table.columns if col != "patientunitstayid"]
            ]

            #ENSURE NO PATIENT IS IN ANY OTHER REGION -- CONRAD
            pid = region_table['stay'].apply(lambda x: x.split("_")[0])
            if pid.isin(patients_visited).astype(int).sum() > 0: 
                print("TRIGGERED", pid[pid.isin(patients_visited)])
                region_table = region_table[~pid.isin(patients_visited)]
            patients_visited = patients_visited | set(pid)

            region_table.to_csv(f"{name}{region}_val.csv", index=False)
            len_val += len(region_table)

        #############################################################
        # Create and save test listfiles
        patients_visited = set()
        for region, region_group in merged_test.groupby("region"):
            region = region.lower()
            table = comp_table[
                comp_table["patientunitstayid"].isin(region_group["patientunitstayid"])
            ]
            region_table = table[
                [col for col in table.columns if col != "patientunitstayid"]
            ]


            #ENSURE NO PATIENT IS IN ANY OTHER REGION -- CONRAD
            pid = region_table['stay'].apply(lambda x: x.split("_")[0])
            if pid.isin(patients_visited).astype(int).sum() > 0: 
                print("TRIGGERED", pid[pid.isin(patients_visited)])
                region_table = region_table[~pid.isin(patients_visited)]
            patients_visited = patients_visited | set(pid)


            region_table.to_csv(f"{name}{region}_test.csv", index=False)
            len_test += len(region_table)

    print("\n")
    print("Training samples:")
    print(len_tr)
    print("Validation samples:")
    print(len_val)
    print("Test samples:")
    print(len_test)


# Append to IHM listfile iff file has atleast 48 hrs of records and a non-nan label and atleast 1 record <= 48 hrs
def append_ihm(filename, min_itemoffset, los, expired, hexpired, ihm_listfile):
    HOURS_MINIMUM = 48
    if los >= HOURS_MINIMUM and hexpired in ["Expired", "Alive"] and min_itemoffset <= HOURS_MINIMUM:
        ihm_listfile['stay'].append(filename)
        ihm_listfile['y_true'].append(1 if hexpired == "Expired" else 0)

# Append to Decomp listfile iff length-of-stay >= 5 hrs and will append hourly window
def append_decomp(filename, min_itemoffset, los, expired, hexpired, decomp_listfile):
    HOURS_MINIMUM = 5
    if los >= HOURS_MINIMUM and expired in ["Expired", "Alive"]:

        for t in range(max(HOURS_MINIMUM,int(min_itemoffset)), int(los)+1):
            decomp_listfile['stay'].append(filename)
            decomp_listfile['period_length'].append(t)
            if expired == "Expired" and t + 24 >= los:
                decomp_listfile['y_true'].append(1)
            else:
                decomp_listfile['y_true'].append(0)
        
# Append to LoS listfile iff length-of-stay >= 5 hrs and will append hourly window
def append_los(filename, min_itemoffset, los, expired, hexpired, los_listfile):
    HOURS_MINIMUM = 5
    if los >= HOURS_MINIMUM:
        for t in range(max(HOURS_MINIMUM,int(min_itemoffset)), int(los)+1):
            los_listfile['stay'].append(filename)
            los_listfile['period_length'].append(t)
            los_listfile['y_true'].append(los - t)

# Append to pheno listfile
def append_pheno(filename, los, pheno_listfile, pheno, unitstayid):
    pheno_listfile['stay'].append(filename)
    pheno_listfile['period_length'].append(los)
    try:
        pheno_res = pheno.loc[unitstayid]
        for col in get_phenotype_order():
            pheno_listfile[col].append(pheno_res[col])
    except KeyError:
        for col in get_phenotype_order():
            pheno_listfile[col].append(0)


def create_dir(task_path, task, listfile, splits=None):
    fullname = {'pheno': 'phenotyping_split/',
                'ihm': 'in-hospital-mortality_split/',
                'decomp': 'decompensation_split/',
                'los': 'length-of-stay_split/',
            }
    new_dir = os.path.join(task_path, fullname[task])
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    
    new_dir_ttv = os.path.join(task_path, fullname[task].split("_")[0]+"/")
    if not os.path.exists(new_dir_ttv):
        os.mkdir(new_dir_ttv)

    #Global listfile
    COL = ['stay', 'period_length', 'y_true']
    if task == 'ihm':
        COL.remove('period_length')
    elif task == 'pheno':
        COL.remove('y_true')
        COL += get_phenotype_order()

    global_listfile = pd.DataFrame(listfile)[COL]
    global_listfile.to_csv(os.path.join(task_path, 'root/', f'{task}_listfile.csv'), index=False)

    #Train-val-test split files
    if splits is not None:
        test_union_val = splits['test'] | splits['val']

        mask = ~(global_listfile['stay'].apply(lambda x: x.split("_")[0])).isin(test_union_val)
        global_listfile[mask].to_csv(os.path.join(new_dir_ttv, f'train_listfile.csv'), index=False)

        mask = (global_listfile['stay'].apply(lambda x: x.split("_")[0])).isin(splits['test'])
        global_listfile[mask].to_csv(os.path.join(new_dir_ttv, f'test_listfile.csv'), index=False)

        mask = (global_listfile['stay'].apply(lambda x: x.split("_")[0])).isin(splits['val'])
        global_listfile[mask].to_csv(os.path.join(new_dir_ttv, f'val_listfile.csv'), index=False)

def populate_root(eicu_path, task_list, splits, patients, mapper, min_records=15):
    #listfiles created as dictionaries
    append_functions = {
        'ihm': append_ihm,
        'pheno': append_pheno,
        'decomp': append_decomp,
        'los': append_los
    }

    all_ihm_listfile = {'stay': [], 'y_true':[]}
    all_decomp_listfile = {'stay': [], 'period_length':[], 'y_true':[]}
    all_los_listfile = {'stay': [], 'period_length':[], 'y_true':[]}
    all_pheno_listfile =  {'stay': [], 'period_length':[]}
    for col in get_phenotype_order():
        all_pheno_listfile[col] = []

    listfiles = {
        'ihm': all_ihm_listfile,
        'pheno': all_pheno_listfile,
        'decomp': all_decomp_listfile,
        'los': all_los_listfile
    }

    pheno = None
    do_pheno = False
    if "pheno" in task_list:
        print("Generating phenotyping labels with >= 1 phenotype...")
        pheno = get_pheno_listfile(eicu_path)
        pheno = pheno.set_index('stay')
        task_list.remove('pheno')
        do_pheno = True 

    print("Task directories and labels creating:", task_list)
    print("70-15-15 train-validation-test split?", splits is not None)
    n = len(patients)

    p = patients.copy()
    p['hospitaldischargestatus']=p['hospitaldischargestatus'].fillna('')
    p['unitdischargestatus']=p['unitdischargestatus'].fillna('')
    p1=p[(p['unitdischargestatus']=='Expired') & (p['hospitaldischargestatus']!='Expired')]
    p2=p[(p['unitdischargestatus']=='') & (p['hospitaldischargestatus']=='Alive')]
    p=pd.concat([p1,p2])
    em = dict(zip(mapper['unitstayid'], mapper['episode']))
    def get_ts(x):
        if x in em:
            return em[x]
        return ''
    p['stay'] = p['patientunitstayid'].apply(get_ts)
    contradict=set(p['stay'])

    for i,(stay, min_time, los, file_length, status, hstatus) in enumerate(
            zip(patients['stay'], patients['min_time'], patients['los'], patients['file_length'], patients['status'], patients['hstatus'])
        ):
        print(f"Completed {i/n*100:.2f}%...", end="\r")

        if file_length < min_records or min_time > los or np.isnan(float(los)):
            continue

        for task in task_list:
            if task in ['ihm','decomp'] and filename in contradict:
                continue
            append_functions[task](stay, min_time, los, status, hstatus, listfiles[task])

        if do_pheno:
            unitstayid = mapper.loc[stay]['unitstayid']
            append_pheno(stay, los, listfiles['pheno'], pheno, unitstayid)

    if do_pheno:
        task_list.append('pheno')  
    
    #To pandas
    for key in listfiles:
        listfiles[key] = pd.DataFrame(listfiles[key]).sample(frac=1)

    gc.collect()
    
    return listfiles
