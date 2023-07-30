import pandas as pd
from sklearn.model_selection import train_test_split

from privacyflow.configs import path_configs

job_to_wage = {
    'Software Engineer': 125,
    'Doctor': 225,
    'Sales Representative': 50,
    'Nurse': 90,
    'Teacher': 90,
    'Accountant': 85,
    'Scientist': 100,
    'Lawyer': 160,
    'Engineer': 107,
    'Salesperson': 50
}

gender_dict = {
    'male': 0,
    'female': 1
}

bmi_dict = {
    'normal': 0,
    'normal weight': 0,
    'overweight': 1,
    'obese': 2
}

sleep_disorder_dict = {
    'None': 0,
    'Sleep Apnea': 1,
    'Insomnia': 2
}

kat_to_sleep_disorder = {
    0: 'None',
    1: 'Sleep Apnea',
    2: 'Insomnia'
}


def preprocess_sleep_data():
    df = pd.read_csv(path_configs.SLEEP_DATA)

    # convert categorial variables
    df['Gender'] = [gender_dict.get(gender.lower()) for gender in df['Gender']]
    df['BMI Category'] = [bmi_dict.get(bmi.lower()) for bmi in df['BMI Category']]
    df['Avg Wage'] = [job_to_wage.get(job, 60) for job in df['Occupation']]
    df['Blood Pressure - systolic'] = [int(bp[:3]) for bp in df['Blood Pressure']]
    df['Blood Pressure - diastolic'] = [int(bp[-2:]) for bp in df['Blood Pressure']]
    df['Sleep Disorder'] = [sleep_disorder_dict.get(sdo, 0) for sdo in df['Sleep Disorder']]
    df = df.drop(columns=['Blood Pressure', 'Occupation', 'Person ID'])

    # other columns
    df['Daily Steps'] = [steps / 1000 for steps in df['Daily Steps']]
    df['Physical Activity Level'] = [act / 10 for act in df['Physical Activity Level']]


    # Train Val Test Split
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['Sleep Disorder'])
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42, stratify=df_train['Sleep Disorder'])

    # Save Files
    df_train.to_csv(path_configs.SLEEP_DATA_PREP_TRAIN, index=False)
    df_val.to_csv(path_configs.SLEEP_DATA_PREP_VAL, index=False)
    df_test.to_csv(path_configs.SLEEP_DATA_PREP_TEST, index=False)