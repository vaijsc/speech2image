'''
This file contains functions to download text prompts of various datasets.
The datasets include PixArt-alpha, diffusiondb, pd12m, laion_12m, and journeydb.
The functions in this file use the Hugging Face Datasets library to load the datasets.
'''

from datasets import load_dataset
from urllib.request import urlretrieve
import pandas as pd

def pixart_alpha():
    # PixArt-alpha
    ds = load_dataset("PixArt-alpha/SAM-LLaVA-Captions10M")
    return ds

def diffusiondb():
    # diffusiondb
    # Download the parquet table
    table_url = 'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata-large.parquet'
    urlretrieve(table_url, '../../../data/raw/text_prompts/diffusiondb-metadata-large.parquet')
    # Read the table using Pandas
    metadata_df = pd.read_parquet('../../../data/raw/text_prompts/diffusiondb-metadata-large.parquet')
    return metadata_df

def pd12m():
    # pd12m
    ds = load_dataset("Spawning/PD12M")
    return ds

def laion_12m():
    # laion 12m
    ds = load_dataset("dclure/laion-aesthetics-12m-umap")
    return ds

def journeydb():
    # journeydb
    ds = load_dataset("JourneyDB/JourneyDB", token="hf_caTwdABsJhoNQXbZeUurdicrbiWsbL", streaming=True)
    return ds

if __name__ == "__main__":
    # pixart_alpha_ds = pixart_alpha()
    # diffusiondb_df = diffusiondb()
    # pd12m_ds = pd12m()
    # laion_12m_ds = laion_12m()
    journeydb_ds = journeydb()

    # Print the structure of the JourneyDB dataset as an example
    first_example = next(iter(journeydb_ds['train']))
    print("Dataset structure:")
    print(first_example.keys())
    for key in first_example:
        print(f"\n{key}:")
        print(first_example[key])
