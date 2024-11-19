import hashlib

import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    if df[id_column].dtype == np.object:
        df[id_column] = df[id_column].apply(lambda x: sum(ord(char)) for char in x)
        df['sample'] = df[id_column].apply(lambda x: 'train' if x % 100 < training_frac * 100 else 'test')
    else:
        df['sample'] = df[id_column].apply(lambda x: 'train' if x % 100 < training_frac * 100 else 'test')
    return df

