# This file is the utils.py file developed by Sheikhalishahi et al.
# Repo: https://github.com/mostafaalishahi/eICU_Benchmark_updated
# We modified the file to preprocess additional features (capillary refill), 
# ignore other features (like age and gender), and use different exclusion criteria.

from __future__ import absolute_import
from __future__ import print_function

import os
import pandas as pd
import numpy as np
import sys
import shutil


def dataframe_from_csv(path, header=0, index_col=False):
    return pd.read_csv(path, header=header, index_col=index_col)

#   Different features (+ Cap Refill)
var_to_consider = ['glucose', 'Invasive BP Diastolic', 'Invasive BP Systolic',
                   'O2 Saturation', 'Respiratory Rate', 'Motor', 'Eyes', 'MAP (mmHg)',
                   'Heart Rate', 'GCS Total', 'Verbal', 'pH', 'FiO2', 'Temperature (C)', 'Capillary Refill']


#Filter on useful column for this benchmark
#   only concerned with weight and height features
def filter_patients_on_columns_model(patients):
    columns = ['patientunitstayid', 'admissionheight', 'hospitaladmitoffset', 'admissionweight',
               'hospitaldischargestatus', 'unitdischargeoffset', 'unitdischargestatus']
    return patients[columns]

#Select unique patient id
def cohort_stay_id(patients):
    cohort = patients.patientunitstayid.unique()
    return cohort

#   Removed unused demographic transformation functions

#Convert hospital/unit discharge status into numbers
h_s_map = {'Expired': 1, 'Alive': 0, '': 2, 'NaN': 2}
def transform_hospital_discharge_status(status_series):
    global h_s_map
    return {'hospitaldischargestatus': status_series.fillna('').apply(
        lambda s: h_s_map[s] if s in h_s_map else h_s_map[''])}

def transform_unit_discharge_status(status_series):
    global h_s_map
    return {'unitdischargestatus': status_series.fillna('').apply(
        lambda s: h_s_map[s] if s in h_s_map else h_s_map[''])}


## Extract the root data

#Extract data from patient table
#   Removed unused demographic transformation functions
def read_patients_table(eicu_path, output_path):
    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv.gz'), index_col=False)
    pats = filter_one_unit_stay(pats)
    pats = filter_patients_on_columns(pats)
    pats = filter_patients_on_age(pats)
    pats.update(transform_hospital_discharge_status(pats.hospitaldischargestatus))
    pats.update(transform_unit_discharge_status(pats.unitdischargestatus))
    pats.to_csv(os.path.join(output_path, 'all_stays.csv'), index=False)
    pats = filter_patients_on_columns_model(pats)
    return pats

#Select unique patient id
def cohort_stay_id(patients):
    cohort = patients.patientunitstayid.unique()
    return cohort

#filter on adult patients
#   just eliminate < 18 years old
def filter_patients_on_age(patient, min_age=18, max_age=100):
    patient.loc[patient['age'] == '> 89', 'age'] = 90
    patient[['age']] = patient[['age']].fillna(-1)
    patient[['age']] = patient[['age']].astype(int)
    patient = patient.loc[(patient.age >= min_age) & (patient.age <= max_age)]
    return patient

#filter those having just one stay in unit
#   Filter those having just one stay in the ICU during a hospital admission. 
# That is, groupby 'patienthealthsystemstayid' instead of 'uniquepid'
def filter_one_unit_stay(patients):
    cohort_count = patients.groupby(by='patienthealthsystemstayid').count()
    cohort_max = patients.groupby(by='patienthealthsystemstayid')['unitvisitnumber'].max()
    index_cohort = cohort_count[(cohort_count['patientunitstayid'] == 1) & (cohort_max == 1)].index
    patients = patients[patients['patienthealthsystemstayid'].isin(index_cohort)]
    return patients

#Filter on useful columns from patient table
#   removed hospitaladmityear and other unused demographic features
def filter_patients_on_columns(patients):
    columns = ['patientunitstayid', 'hospitaldischargeyear', 'hospitaldischargeoffset',
               'admissionheight', 'hospitaladmitoffset', 'admissionweight', 'age',
               'hospitaldischargestatus', 'unitdischargeoffset', 'unitdischargestatus']
    return patients[columns]

#Write the selected cohort data from patient table into pat.csv for each patient
def break_up_stays_by_unit_stay(pats, output_path, stayid=None, verbose=1):
    unit_stays = pats.patientunitstayid.unique() if stayid is None else stayid
    nb_unit_stays = unit_stays.shape[0]
    for i, stay_id in enumerate(unit_stays):
        if verbose:
            sys.stdout.write('\rStayID {0} of {1}...'.format(i + 1, nb_unit_stays))
        dn = os.path.join(output_path, str(stay_id))
        try:
            os.makedirs(dn)
        except:
            pass

        pats.loc[pats.patientunitstayid == stay_id].sort_values(by='hospitaladmitoffset').to_csv(
            os.path.join(dn, 'pats.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')

##   Here we deal with the nurseAssessment table
#Select the useful columns from  nurseAssessment table
def filter_na_on_columns(na):
    columns = ['patientunitstayid', 'nurseassessoffset', 'cellattribute', 'cellattributevalue']
    return na[columns]

#Rename the columns in order to have a unified name
def rename_na_columns(na):
    na.rename(index=str, columns={'nurseassessoffset': "itemoffset", "cellattribute": "itemname",
                                   "cellattributevalue": "itemvalue"}, inplace=True)
    return na

#   Select the Capillary Refill measurement from nurseAssessment table
def item_name_selected_from_na(na, items):
    na = na[na['itemname'].isin(items)]
    return na

#   Check if the Capillary Refill measurement is valid
def check_na(x):
    if x not in ["normal", "< 2 seconds", "> 2 seconds"]:
        x = np.nan
    return x
def check_itemvalue_na(df):
    df['itemvalue'] = df['itemvalue'].apply(lambda x: check_na(x))
    return df


#   Encodes the two outcomes of a capillary refill test. 
def encode_cr_result(na):
    na["itemvalue"] = na["itemvalue"].replace({"normal": 0, "< 2 seconds": 0, "> 2 seconds": 1})
    return na

#   Removes hands and feet data points  
def remove_null_in_na(na):
    mask = ~na['itemvalue'].isnull()
    return na[mask]

def read_na_table(eicu_path):
    na = dataframe_from_csv(os.path.join(eicu_path, 'nurseAssessment.csv.gz'), index_col=False)
    na = filter_na_on_columns(na)
    na = rename_na_columns(na)
    items = ["Capillary Refill"]
    na = item_name_selected_from_na(na, items)
    na = check_itemvalue_na(na)
    na = encode_cr_result(na)
    na = remove_null_in_na(na)
    return na

#Write the nc values of each patient into a na.csv file
def break_up_na_by_unit_stay(nurseAssess, output_path, stayid=None, verbose=1):
    unit_stays = nurseAssess.patientunitstayid.unique() if stayid is None else stayid
    nb_unit_stays = unit_stays.shape[0]
    for i, stay_id in enumerate(unit_stays):
        if verbose:
            sys.stdout.write('\rStayID {0} of {1}...'.format(i + 1, nb_unit_stays))
        dn = os.path.join(output_path, str(stay_id))
        try:
            os.makedirs(dn)
        except:
            pass

        nurseAssess.loc[nurseAssess.patientunitstayid == stay_id].sort_values(by='itemoffset').to_csv(
            os.path.join(dn, 'na.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')

## END

## Here we deal with lab table
#Select the useful columns from lab table
def filter_lab_on_columns(lab):
    columns = ['patientunitstayid', 'labresultoffset', 'labname', 'labresult']
    return lab[columns]

#Rename the columns in order to have a unified name
def rename_lab_columns(lab):
    lab.rename(index=str, columns={"labresultoffset": "itemoffset",
                                   "labname": "itemname", "labresult": "itemvalue"}, inplace=True)
    return lab

#Select the lab measurement from lab table
def item_name_selected_from_lab(lab, items):
    lab = lab[lab['itemname'].isin(items)]
    return lab

#Check if the lab measurement is valid
def check(x):
    try:
        x = float(str(x).strip())
    except:
        x = np.nan
    return x
def check_itemvalue(df):
    df['itemvalue'] = df['itemvalue'].apply(lambda x: check(x))
    df['itemvalue'] = df['itemvalue'].astype(float)
    return df

#extract the lab items for each patient
def read_lab_table(eicu_path):
    lab = dataframe_from_csv(os.path.join(eicu_path, 'lab.csv.gz'), index_col=False)
    items = ['bedside glucose', 'glucose', 'pH', 'FiO2']
    lab = filter_lab_on_columns(lab)
    lab = rename_lab_columns(lab)
    lab = item_name_selected_from_lab(lab, items)
    lab.loc[lab['itemname'] == 'bedside glucose', 'itemname'] = 'glucose'  # unify bedside glucose and glucose
    lab = check_itemvalue(lab)
    return lab


#Write the available lab items of a patient into lab.csv
def break_up_lab_by_unit_stay(lab, output_path, stayid=None, verbose=1):
    unit_stays = lab.patientunitstayid.unique() if stayid is None else stayid
    nb_unit_stays = unit_stays.shape[0]
    for i, stay_id in enumerate(unit_stays):
        if verbose:
            sys.stdout.write('\rStayID {0} of {1}...'.format(i + 1, nb_unit_stays))
        dn = os.path.join(output_path, str(stay_id))
        try:
            os.makedirs(dn)
        except:
            pass
        lab.loc[lab.patientunitstayid == stay_id].sort_values(by='itemoffset').to_csv(os.path.join(dn, 'lab.csv'),
                                                                                     index=False)
    if verbose:
        sys.stdout.write('DONE!\n')


#Filter the useful columns from nc table
def filter_nc_on_columns(nc):
    columns = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel',
               'nursingchartcelltypevalname', 'nursingchartvalue']
    return nc[columns]

#Unify the column names in order to be used later
def rename_nc_columns(nc):
    nc.rename(index=str, columns={"nursingchartoffset": "itemoffset",
                                  "nursingchartcelltypevalname": "itemname",
                                  "nursingchartcelltypevallabel": "itemlabel",
                                  "nursingchartvalue": "itemvalue"}, inplace=True)
    return nc

#Select the items using name and label
def item_name_selected_from_nc(nc, label, name):
    nc = nc[(nc.itemname.isin(name)) |
            (nc.itemlabel.isin(label))]
    return nc

#Convert fahrenheit to celsius
def conv_far_cel(nc):
    nc['itemvalue'] = nc['itemvalue'].astype(float)
    nc.loc[nc['itemname'] == "Temperature (F)", "itemvalue"] = ((nc['itemvalue'] - 32) * (5 / 9))

    return nc

#Unify the different names into one for each measurement
def replace_itemname_value(nc):
    nc.loc[nc['itemname'] == 'Value', 'itemname'] = nc.itemlabel
    nc.loc[nc['itemname'] == 'Non-Invasive BP Systolic', 'itemname'] = 'Invasive BP Systolic'
    nc.loc[nc['itemname'] == 'Non-Invasive BP Diastolic', 'itemname'] = 'Invasive BP Diastolic'
    nc.loc[nc['itemname'] == 'Temperature (F)', 'itemname'] = 'Temperature (C)'
    nc.loc[nc['itemlabel'] == 'Arterial Line MAP (mmHg)', 'itemname'] = 'MAP (mmHg)'
    nc.loc[nc['itemname'] == 'Non-Invasive BP Mean', 'itemname'] = 'MAP (mmHg)'
    nc.loc[nc['itemname'] == 'Invasive BP Mean', 'itemname'] = 'MAP (mmHg)'
    
    return nc 


#Select the nurseCharting items and save it into nc
def read_nc_table(eicu_path):
    # import pdb;pdb.set_trace()
    nc = dataframe_from_csv(os.path.join(eicu_path, 'nurseCharting.csv.gz'), index_col=False)
    nc = filter_nc_on_columns(nc)
    nc = rename_nc_columns(nc)
    typevallabel = ['Glasgow coma score', 'Heart Rate', 'O2 Saturation', 'Respiratory Rate', 'MAP (mmHg)',
                    'Arterial Line MAP (mmHg)']
    typevalname = ['Non-Invasive BP Systolic', 'Invasive BP Systolic', 'Non-Invasive BP Diastolic',
                   'Invasive BP Diastolic', 'Temperature (C)', 'Temperature (F)',
                   'Invasive BP Mean','Non-Invasive BP Mean']
    nc = item_name_selected_from_nc(nc, typevallabel, typevalname)
    nc = check_itemvalue(nc)
    nc = conv_far_cel(nc)
    replace_itemname_value(nc)
    del nc['itemlabel']
    return nc


#Write the nc values of each patient into a nc.csv file
def break_up_stays_by_unit_stay_nc(nursecharting, output_path, stayid=None, verbose=1):
    unit_stays = nursecharting.patientunitstayid.unique() if stayid is None else stayid
    nb_unit_stays = unit_stays.shape[0]
    for i, stay_id in enumerate(unit_stays):
        if verbose:
            sys.stdout.write('\rStayID {0} of {1}...'.format(i + 1, nb_unit_stays))
        dn = os.path.join(output_path, str(stay_id))
        try:
            os.makedirs(dn)
        except:
            pass

        nursecharting.loc[nursecharting.patientunitstayid == stay_id].sort_values(by='itemoffset').to_csv(
            os.path.join(dn, 'nc.csv'), index=False)
    if verbose:
        sys.stdout.write('DONE!\n')


# Write the time-series data into one csv for each patient
#   There are multiple changes here
# 1. Imputation and binning is not done here 
# 2. truncates timeseries files further to match MIMIC-III benchmark preprocessing
# 3. skips over files that have < 15 records during their stay
# 4. creates new file called timeseries_info.csv which is used later to identify which patients violate task-specific exclusion criteria 
def extract_time_series_from_subject(t_path, eicu_path, tsk_path):
    print("Convert to time series ...")
    print("This will take some hours, as converting time series are done here ...")

    MINIMUM_RECORDS = 15
    column_order = ["itemoffset", "Capillary Refill", "Invasive BP Diastolic", "FiO2", "Eyes",
        "Motor", "GCS Total", "Verbal", "glucose", "Heart Rate", "admissionheight", "MAP (mmHg)", 
        "O2 Saturation", "Respiratory Rate", "Invasive BP Systolic", "Temperature (C)",
        "admissionweight", "pH"]
    
    mapper = dataframe_from_csv('resources/episode_mapper.csv')
    mapper = mapper.set_index('unitstayid')

    pats = dataframe_from_csv(os.path.join(eicu_path, 'patient.csv.gz'), index_col=False)
    pats = pats.set_index('patientunitstayid')
    pats = pats[['unitdischargeoffset', 'unitdischargestatus', 'hospitaldischargestatus']]
        
    with open(os.path.join(tsk_path, 'root/timeseries_info.csv'), "w") as f:
        f.write('stay,min_time,los,file_length,status,hstatus\n')
    with open(os.path.join(tsk_path, 'root/timeseries_info.csv'), "a") as f:
        for i,stay_dir in enumerate(os.listdir(t_path)):
            # import pdb;pdb.set_trace()
            dn = os.path.join(t_path, stay_dir)
            if not os.path.isdir(dn):
                print(dn, os.path.isdir(dn))
                continue
            stay_id = int(stay_dir)
            sys.stdout.write('\rWrite patient {0} / {1}'.format(i,len(os.listdir(t_path))))
            try:
                pat = dataframe_from_csv(os.path.join(t_path, stay_dir, 'pats.csv'))
                lab = dataframe_from_csv(os.path.join(t_path, stay_dir, 'lab.csv'))
                nc = dataframe_from_csv(os.path.join(t_path, stay_dir, 'nc.csv'))
                na = dataframe_from_csv(os.path.join(t_path, stay_dir, 'na.csv'))
                if len(lab) + len(nc) + len(na) == 0:
                    continue
                nclab = pd.concat([nc, lab, na]).sort_values(by=['itemoffset'])
                timeepisode = convert_events_to_timeseries(nclab, variables=var_to_consider)
                df = pd.merge(timeepisode, pat, on='patientunitstayid')
                
                df = check_in_range(df)
                df.to_csv(os.path.join(t_path, stay_dir, 'timeseries.csv'), index=False)

                los = pats.loc[stay_id]['unitdischargeoffset']
                df = truncation(df, los, column_order)
                if df is None or len(df) < MINIMUM_RECORDS:
                    continue 

                #formate timeseries files as "{uniquepid}_episode{visit number}_timeseries.csv"
                new_filename = mapper.loc[stay_id]['episode']
                df.to_csv(os.path.join(tsk_path, 'root/', new_filename), index=False)
                mn = df['itemoffset'].min()
                status = pats.loc[stay_id]['unitdischargestatus']
                hstatus = pats.loc[stay_id]['hospitaldischargestatus']
                f.write(f'{new_filename},{mn},{los/60},{len(df)},{status},{hstatus}\n')

            except Exception as e:
                print(e)
                continue
        print('Converted to time series')

##Convert to time-series

#   Additional preprocessing
def truncation(curr_patient, los, column_order):
    NUM_FEATURES = 17

    #Truncate between [0, LoS]
    curr_patient = curr_patient[np.logical_and(curr_patient['itemoffset'] >= 0, curr_patient['itemoffset'] <= los)]
    curr_patient = curr_patient.reset_index(drop=True)[column_order]
    curr_patient['itemoffset'] = curr_patient['itemoffset'] / 60

    if len(curr_patient) == 0:
        return None

    #Set height and weight to itemoffset 0
    h = float(curr_patient['admissionheight'].max())
    w = float(curr_patient['admissionweight'].max())

    if curr_patient['itemoffset'].min() != 0 and (not np.isnan(h) or not np.isnan(w)):
        curr_patient.loc[-1] = [0]+[np.nan]*NUM_FEATURES
        curr_patient.index = curr_patient.index + 1
        curr_patient.sort_index(inplace=True)
    
    curr_patient['admissionheight'] = [h] + [np.nan]*(len(curr_patient)-1)
    curr_patient['admissionweight'] = [w] + [np.nan]*(len(curr_patient)-1)
    curr_patient = curr_patient[column_order]

    curr_patient = curr_patient[~curr_patient[curr_patient.columns[1:]].isnull().all(axis=1)]

    return curr_patient[column_order].sort_values(by='itemoffset')


#Check the range of each measurment
def check_in_range(df):
    df['Eyes'].clip(0, 5, inplace=True)
    df['GCS Total'].clip(2, 16, inplace=True)
    df['Heart Rate'].clip(0, 350, inplace=True)
    df['Motor'].clip(0, 6, inplace=True)
    df['Invasive BP Diastolic'].clip(0, 375, inplace=True)
    df['Invasive BP Systolic'].clip(0, 375, inplace=True)
    df['MAP (mmHg)'].clip(14, 330, inplace=True)
    df['Verbal'].clip(1, 5, inplace=True)
    df['admissionheight'].clip(100, 240, inplace=True)
    df['admissionweight'].clip(30, 250, inplace=True)
    df['glucose'].clip(33, 1200, inplace=True)
    df['pH'].clip(6.3, 10, inplace=True)
    df['FiO2'].clip(15, 110, inplace=True)
    df['O2 Saturation'].clip(0, 100, inplace=True)
    df['Respiratory Rate'].clip(0, 100, inplace=True)
    df['Temperature (C)'].clip(26, 45, inplace=True)
    df['Capillary Refill'].clip(0, 1, inplace=True)
    return df

#Read each patient nc, lab and demographics and put all in one csv
def convert_events_to_timeseries(events, variable_column='itemname', variables=[]):
    metadata = events[['itemoffset', 'patientunitstayid']].sort_values(
        by=['itemoffset', 'patientunitstayid']).drop_duplicates(keep='first').set_index('itemoffset')

    timeseries = events[['itemoffset', variable_column, 'itemvalue']].sort_values(
        by=['itemoffset', variable_column, 'itemvalue'], axis=0).drop_duplicates(subset=['itemoffset', variable_column],
                                                                                 keep='last')
    timeseries = timeseries.pivot(index='itemoffset', columns=variable_column, values='itemvalue').merge(metadata,
                                                                                                         left_index=True,
                                                                                                         right_index=True).sort_index(
        axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries

#Delete folders without timeseries file
def delete_wo_timeseries(t_path):
    # import pdb;pdb.set_trace()
    for stay_dir in os.listdir(t_path):
        dn = os.path.join(t_path, stay_dir)
        try:
            stay_id = int(stay_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue
        try:
            sys.stdout.flush()
            if not os.path.isfile(os.path.join(dn, 'timeseries.csv')):
                shutil.rmtree(dn)
        except:
            continue
    print('DONE deleting')

#Write all the extracted data into one csv file
def all_df_into_one_df(output_path):
    # import pdb;pdb.set_trace()
    all_filenames = []
    unit_stays = pd.Series(os.listdir(output_path))
    unit_stays = list((filter(str.isdigit, unit_stays)))
    for stay_id in (unit_stays):
        df_file = os.path.join(output_path, str(stay_id), 'timeseries.csv')
        all_filenames.append(df_file)

    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv.to_csv(os.path.join(output_path, 'all_data.csv'), index=False)
