#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import gradio as gr
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth',500)  


# In[9]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import gensim.downloader as api
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from gensim.utils import simple_preprocess
from concurrent.futures import ThreadPoolExecutor
import chardet
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from UI_semanticMatching import *
matcher = MatchingFunction()
from UI_input_output import read_clean_model


# In[11]:


# Backend logic: accepts two files, processes them, returns a new DataFrame
def process_two_csvs(file_path_input, file_path_mdr):
    if file_path_input is None or file_path_mdr is None:
        return pd.DataFrame()  # Return empty DataFrame if either is missing

    # Read both files into DataFrames with their own encoding
    with open(file_path_input, 'rb') as f:
        result = chardet.detect(f.read())
    encoding_input = result['encoding']
    
    with open(file_path_mdr, 'rb') as f:
        result = chardet.detect(f.read())
    encoding_mdr = result['encoding']
    
    input_df = pd.read_csv(file_path_input, encoding=encoding_input)
    print(len(input_df))
    mdr_df = pd.read_csv(file_path_mdr, encoding=encoding_mdr)
    print(len(mdr_df))

    # clean and run through models
    result_df = read_clean_model(input_df, mdr_df)

    return result_df

# Gradio interface with two file inputs
csv_input = gr.File(label="Upload Input File (make sure the file contains at least two columns: variable and description)",
                    file_types=[".csv"])
csv_mdr = gr.File(label="Upload MDR File (make sure the file contains at least two columns: name and definition)", 
                    file_types=[".csv"])
output_table = gr.Dataframe(label="Processed Output Table")

# Launch the app
demo = gr.Interface(
    fn=process_two_csvs,
    inputs=[csv_input, csv_mdr],
    outputs=output_table,
    title="Semantic Matching between Input and MDR",
    description="Upload files."
)

demo.launch(share=True)


# In[ ]:




