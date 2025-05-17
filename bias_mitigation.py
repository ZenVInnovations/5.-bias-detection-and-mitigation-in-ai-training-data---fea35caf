import pandas as pd
from sklearn.utils import resample

def oversample_minority(data, target, minority_class):
    '''Oversample the minority class in the dataset.'''
    df_majority = data[data[target] != minority_class]
    df_minority = data[data[target] == minority_class]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    return pd.concat([df_majority, df_minority_upsampled])

def undersample_majority(data, target, majority_class):
    '''Undersample the majority class in the dataset.'''
    df_majority = data[data[target] == majority_class]
    df_minority = data[data[target] != majority_class]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    return pd.concat([df_majority_downsampled, df_minority])
