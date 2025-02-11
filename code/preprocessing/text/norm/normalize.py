'''normalize text in 4 prompt datasets'''

from tqdm import tqdm
import pandas as pd
from cleaners import english_cleaners


if __name__ == '__main__':
    # read csv file 
    print('reading file...')
    df = pd.read_csv('/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/text_prompts/shorten_csvs/diffusiondb_with_text.csv')
    print(f'df shape: {df.shape}')

    # normalize text 
    print('normalizing text...')
    normalized_prompts = []
    for _, sample in tqdm(df.iterrows(), total=df.shape[0], desc='Normalizing prompts'):
        image_name, prompt = sample['image_name'], sample['prompt']
        try:
            norm_prompt = english_cleaners(prompt)
            normalized_prompts.append(norm_prompt)
        except Exception as e:
            print(f"Error normalizing prompt: {e}")
            normalized_prompts.append(None)  # Append None for failed normalizations

    # Create a new DataFrame with the normalized prompts and corresponding data
    normalized_df = df.copy()
    normalized_df['normalized_prompt'] = normalized_prompts
    normalized_df = normalized_df[normalized_df['normalized_prompt'].notnull()]  # Keep only rows with valid normalized prompts
    print(f'normalized_df shape: {normalized_df.shape}')

    # save to file
    print('saving file...')
    normalized_df.to_csv('/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/text_prompts/norm/diffusiondb_normalized.csv', index=False)
