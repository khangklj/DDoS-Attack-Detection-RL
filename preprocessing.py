import numpy as np
from sklearn.preprocessing import LabelEncoder

SEPERATOR = "================="

def preprocess(df):   
    """
    Preprocess the given DataFrame by dropping unnecessary columns, removing inf and NaN, removing duplicated rows, encoding Class column to 0 or 1, and encoding Label column to 0,1,2,... for each unique value of df.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be preprocessed

    Returns
    -------
    tuple
        A tuple of two dictionaries, where the first dictionary is a mapping of the Class column to 0 or 1, 
        and the second dictionary is a mapping of the Label column to 0,1,2,... for each unique value of df.
    """
    drop_columns = [ # drop unnecessary columns
    "Unnamed: 0",
    ]
    print("Drop columns: ", drop_columns)
    df.columns = df.columns.str.strip()
    df.drop(columns=drop_columns, inplace=True)
    print("Data shape: ", df.shape)

    print(SEPERATOR)

    # Remove inf and NaN
    print("Drop NaN: ")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(df.isna().sum())
    df.dropna(inplace=True)
    print("Data shape: ", df.shape)

    print(SEPERATOR)

    # Remove duplicated
    print("Duplicated: ", df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    print("Data shape: ", df.shape)

    print(SEPERATOR)

    # Encode Class column to 0 or 1
    print("Encode Class column")
    binary_le = LabelEncoder()
    df['Class'] = binary_le.fit_transform(df['Class'])
    binary_le_mapping = dict(zip(binary_le.classes_, binary_le.transform(binary_le.classes_)))
    for key, value in binary_le_mapping.items():
        print(f"{key}: {value}")
    print("Data shape: ", df.shape)

    print(SEPERATOR)

    # Encode Label column to 0,1,2,... for each unique value of df
    print("Encode Label column")
    multi_le = LabelEncoder()
    df['Label'] = multi_le.fit_transform(df['Label'])
    multi_le_mapping = dict(zip(multi_le.classes_, multi_le.transform(multi_le.classes_)))
    for key, value in multi_le_mapping.items():
        print(f"{key}: {value}")
    print("Data shape: ", df.shape)

    return binary_le_mapping, multi_le_mapping