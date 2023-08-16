import json
import pandas as pd
import pickle
import os

from os.path import join as pj

def read_df(path: str, convert_index=True) -> pd.DataFrame:
    """
    Read single df. 
    If convert_index is True, this function will check if the index is RangeIndex, 
    and if yes, the index will be converted to I{} format string index. 

    Function coded by Ruibo Zhang.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, engine='fastparquet')
    elif path.endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
    elif path.endswith(".tsv"):
        df = pd.read_table(path, index_col=0)
    elif path.endswith(".pickle") or path.endswith('.pkl'):
        with open(path, 'rb') as f:
            df = pickle.load(f)
    elif path.endswith(".json"):
        dict = json.load(path)
        df = pd.DataFrame(dict)
    elif path.endswith(".txt"):
        df = pd.read_table(path, index_col=0)
    else:
        raise ValueError("Not supported format: " + path)

    if convert_index:
        if isinstance(df.index, pd.RangeIndex):
            df.index = ["I{}".format(ii) for ii in range(len(df))]
        if isinstance(df.columns, pd.RangeIndex):
            df.columns = ["I{}".format(ii) for ii in range(df.shape[1])]

    return df

def data(cwd):

    """
    Reading the data.
    """
    f_list = [pj(cwd, x) for x in os.listdir(cwd)]

    # In the raw data we have the metadata, the SMILES and the information of the compounds (including target)
    f_data = [f for f in f_list if f.endswith(".txt")]
    raw_data = read_df(f_data[0])

    # In fp we have the fingerprints of ECFP4. 
    f_fp = [f for f in f_list if f.endswith("ECFP4.parquet")]
    fp_df = read_df(f_fp[0])

    # In distances we have the precomputed distances
    distance = read_df(pj(cwd, "{}_{}_dist.parquet".format("ecfp4", "tanimoto")))
    # print(f_data[0])

    # Let's reindex everything. 
    index = raw_data.index
    raw_data = raw_data.reindex(index)
    fp_df = fp_df.reindex(index)
    distance = distance.reindex(index)[index]
    return raw_data, distance