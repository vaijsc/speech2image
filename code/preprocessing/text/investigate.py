import pandas as pd
import time


def pixart():
    # load csv
    pixart_df = pd.read_csv('../../../data/processed/text_prompts/shorten_csvs/pixart.csv')
    # check for nan value in prompt column and save new csv
    pixart_df[pixart_df['prompt'].notnull()].to_csv('../../../data/processed/text_prompts/shorten_csvs/pixart_not_nan.csv', index=False)

    # check for special characters: 0 samples 
    start = time.time()
    df_not_nan = pd.read_csv('../../../data/processed/text_prompts/shorten_csvs/pixart_not_nan.csv')
    end = time.time()
    print(f'it takes {(end - start) / 1000}s to load csv file')
    has_special = df_not_nan['prompt'].str.contains(f'[{special_chars}]', regex=True)

    print(f"Number of prompts with special characters: {has_special.sum()}")
    print(f"Percentage of prompts with special characters: {(has_special.sum() / len(df_not_nan)) * 100:.2f}%")

    # Display some examples of prompts with special characters
    print("\nExample prompts with special characters:")
    print(df_not_nan[has_special]['prompt'].head())
    
    
def laion(): 
    # load csv: shape: (12096809, 3), header: ['URL', 'prompt', 'AESTHETIC_SCORE'] 
    laion_df = pd.read_csv('../../../data/processed/text_prompts/shorten_csvs/laion.csv')
    # check for nan value
    null_count = laion_df['prompt'].isnull().sum()
    print(f"Number of null prompts: {null_count}") # 0
    print(f"Percentage of null prompts: {(null_count / len(laion_df)) * 100:.2f}%")
    laion_df_not_nan = laion_df[laion_df['prompt'].notnull()]

    # check for special characters
    has_special_laion = laion_df_not_nan['prompt'].str.contains(f'[{special_chars}]', regex=True)
    print(f"Number of prompts with special characters in laion: {has_special_laion.sum()}")
    print(f"Percentage of prompts with special characters in laion: {(has_special_laion.sum() / len(laion_df_not_nan)) * 100:.2f}%")

    # Display some examples of prompts with special characters
    print("\nExample prompts with special characters in laion:")
    print(laion_df_not_nan[has_special_laion]['prompt'].head())
    laion_df_not_nan.to_csv('../../../data/processed/text_prompts/shorten_csvs/laion_no_special.csv', index=False)


def rm_null_spec(dataset_name):
    df = pd.read_csv(f'../../../data/processed/text_prompts/shorten_csvs/{dataset_name}.csv')
    # check for nan value
    null_count = df['prompt'].isnull().sum()
    print(f"Number of null prompts: {null_count}") 
    print(f"Percentage of null prompts: {(null_count / len(df)) * 100:.2f}%")
    df_not_nan = df[df['prompt'].notnull()]
    
    # save df with not nan values
    df_not_nan.to_csv(f'../../../data/processed/text_prompts/shorten_csvs/{dataset_name}_not_nan.csv', index=False)

    # check for special characters
    has_special = df_not_nan['prompt'].str.contains(f'[{special_chars}]', regex=True)
    print(f"Number of prompts with special characters: {has_special.sum()}")
    print(f"Percentage of prompts with special characters: {(has_special.sum() / len(df_not_nan)) * 100:.2f}%")

    # Display some examples of prompts with special characters
    print("\nExample prompts with special characters:")
    print(df_not_nan[has_special]['prompt'].head())

    # save df with no special characters
    breakpoint()
    df_not_nan.to_csv(f'../../../data/processed/text_prompts/shorten_csvs/{dataset_name}_no_special.csv', index=False)


def remove_all_nums_samples():
    # remove prompts that only contains numbers (diffusiondb has some): 
    df = pd.read_csv('/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/text_prompts/shorten_csvs/diffusiondb_no_special.csv')
    # Keep samples where the prompt contains at least one alphabetical character (a to z)
    df_filtered = df[df['prompt'].str.contains(r'[a-zA-Z]', regex=True)]

    # Identify samples that do not meet the criteria
    df_invalid = df[~df['prompt'].str.contains(r'[a-zA-Z]', regex=True)]
    
    # Print out samples that do not meet the criteria
    print("\nSamples that do not meet the criteria (only numbers or empty prompts):")
    print(df_invalid)

    print(f'shape: {df_filtered.shape}')
    # Save the filtered DataFrame to a new CSV file
    df_filtered.to_csv('/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/text_prompts/shorten_csvs/diffusiondb_with_text.csv', index=False)

    
if __name__ == '__main__':
    special_chars = '[@_!#$%^&*()<>?/\|}{~:]'
    # pixart()
    # laion() 
    # rm_null_spec("pd12m")
    remove_all_nums_samples()