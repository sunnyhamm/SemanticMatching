import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import gensim.downloader as api
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from gensim.utils import simple_preprocess
from concurrent.futures import ThreadPoolExecutor
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import functions from semanticMatching.py, make this semanticMatching.py is located in the same directory where this script is run
from UI_semanticMatching import *
matcher = MatchingFunction()

def read_clean_model(input_df, mdr_df):

    input_columns_to_check = ["Legacy Variable", "Variable Name *", "Description *"]
    mdr_columns_to_check = ["name", "definition"]

    # drop missing, drop dups
    input_df = input_df.dropna(subset=input_columns_to_check)
    mdr_df = mdr_df.dropna(subset=mdr_columns_to_check)
    
    input_df = input_df.drop_duplicates(subset=["Variable Name *"])
    mdr_df = mdr_df.drop_duplicates(subset=["name"])
    
    # Remove place holders like TBD
    
    placeholder_vars = ['tbd'] # lower case
    
    for var in placeholder_vars:
        for col in mdr_columns_to_check:
            to_drop = mdr_df[mdr_df[col].str.lower()==var].index
            mdr_df = mdr_df.drop(to_drop)
    
    for var in placeholder_vars:
        for col in input_columns_to_check:
            to_drop = input_df[input_df[col].str.lower()==var].index
            input_df = input_df.drop(to_drop)

    # removed derived
    derived_data_type_list = ['_DVAL', '_DSUM']
    
    mdr_df['derived'] = 'no'
    mask_derived = (mdr_df['name'].str.contains('|'.join(derived_data_type_list), case=False, na=False))
    mdr_df.loc[mask_derived, 'derived'] = 'yes'
    
    input_df['derived'] = 'no'
    mask_derived = (input_df['Legacy Variable'].str.contains('|'.join(derived_data_type_list), case=False, na=False)) | (input_df['Variable Name *'].str.contains('|'.join(derived_data_type_list), case=False, na=False))
    input_df.loc[mask_derived, 'derived'] = 'yes'

    ## Model learning ...
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    # Encode the existing descriptions from the MDR dataset into embeddings
    definitionEmbeddingsB = model.encode(input_df['Description *'].tolist(), convert_to_tensor=True).to(device)
    definitionEmbeddingsA = model.encode(mdr_df["definition"].tolist(), convert_to_tensor=True).to(device)
    
    # Initialize a list to store results
    results = []
    
    for i, embeddingA in enumerate(definitionEmbeddingsA):
        similarities = util.pytorch_cos_sim(embeddingA, definitionEmbeddingsB)
        most_similar_idx = similarities.argmax().item()
        similarity_score = round(similarities[0][most_similar_idx].item() * 100, 1)
        result = {
            "input_name": input_df.iloc[most_similar_idx][["Legacy Variable"][0]] if ["Legacy Variable"] and ["Legacy Variable"][0] in input_df.columns else None,
            "input_descr": input_df['Description *'].iloc[most_similar_idx],
            "mdr_name": mdr_df.iloc[i][["name"][0]] if ["name"] and ["name"][0] in mdr_df.columns else None,
            "mdr_descr": mdr_df["definition"].iloc[i],
            "similarity_score": similarity_score
        }
    
        results.append(result)
    
    results_df = pd.DataFrame(results)
    sorted_df = results_df.sort_values(by=['similarity_score'], ascending=[False])
    sorted_df['similarity_score'] = sorted_df['similarity_score'].apply(lambda x: str(x) + '%')
    sorted_df = sorted_df.drop_duplicates(subset=["input_descr", "mdr_descr"]).reset_index(drop=True)

    return sorted_df


if __name__ == '__main__':

    print('something')
    
    # # Load datasets 
    # file_path_input = "data/raw/ABS-MOPS Variables - December 11 2024.xlsm"
    # file_path_mdr = "data/raw/mdr Variables 1.xlsx"
    
    # input_df = pd.read_excel(file_path_input, sheet_name="Data Sheet", header=12).rename(columns={'Unnamed: 3':'Legacy Variable'})
    # mdr_df = pd.read_excel(file_path_mdr)

    # output_df = read_clean_model(input_df, mdr_df)