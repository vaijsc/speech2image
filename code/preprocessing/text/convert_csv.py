'''
This file contains functions to convert various datasets into CSV format.
The datasets include laion, diffusiondb, pixart, and pd12m.
Each function loads the dataset, processes it, and saves image ids 
and corresponding prompts to a CSV file in `data/processed/text_prompts/shorten_csvs`.
'''


from tqdm import tqdm
from datasets import load_dataset
import pandas as pd


def laion():
    # laion
    laion = load_dataset("dclure/laion-aesthetics-12m-umap")
    laion = laion['train']

    # Convert to dataframe with URL, TEXT, AESTHETIC_SCORE columns
    laion_df = pd.DataFrame({
        'URL': [item['URL'] for item in tqdm(laion, desc='Creating DataFrame')],
        'prompt': [item['TEXT'] for item in tqdm(laion, desc='Creating DataFrame')],
        'AESTHETIC_SCORE': [item['AESTHETIC_SCORE'] for item in tqdm(laion, desc='Creating DataFrame')]
    })

    # Save the DataFrame to a CSV file
    laion_df.to_csv('../../../data/processed/text_prompts/shorten_csvs/laion.csv', index=False)


def diffusiondb():
    # diffusiondb
    df = pd.read_parquet('../text_data/diffusiondb-metadata-large.parquet')
    unique_prompts_df = df.drop_duplicates(subset='prompt', keep='first')[['image_name', 'prompt']]
    # save
    unique_prompts_df.to_csv('../../../data/processed/text_prompts/diffusiondb.csv', index=False)

    
def pixart():
    pixart_hf = load_dataset("PixArt-alpha/SAM-LLaVA-Captions10M")
    pixart_hf = pixart_hf['train']
    pixart = pd.DataFrame({
        'key': [item['__key__'] for item in tqdm(pixart_hf, desc='Processing PixArt data')],
        'prompt': [item['txt'] for item in tqdm(pixart_hf, desc='Processing PixArt data')]
    })
    pixart.to_csv('../../../data/processed/text_prompts/pixart.csv', index=False)
    

def pd12m(): 
    ds = load_dataset("Spawning/PD12M")
    ds = ds['train']
    pd_df = pd.DataFrame({
        'id': [item['id'] for item in tqdm(ds, desc='Creating DataFrame')],
        'prompt': [item['caption'] for item in tqdm(ds, desc='Creating DataFrame')]
    })
    pd_df.to_csv('../../../data/processed/text_prompts/pd12m.csv', index=False)

def journeydb():
    # have to download each file in huggingface manually
    ds = load_dataset("JourneyDB/JourneyDB", token="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")
    breakpoint()

if __name__ == '__main__':
    # diffusiondb()
    # laion()
    # pixart()
    # pd12m()
    journeydb()