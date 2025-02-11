import argparse
import os
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from infer import load_model, run_tts

# Fixed paths and configurations
DEFAULT_XTTS_CHECKPOINT = 'checkpoints/model.pth'
DEFAULT_XTTS_CONFIG = 'checkpoints/config.json'
DEFAULT_XTTS_VOCAB = 'checkpoints/vocab.json'
DEFAULT_SPEAKER_CSV = '/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/speech_prompts/0_common_voice/sampled_audio.csv'
DEFAULT_PROMPT_CSV = '/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/text_prompts/shorten_csvs/pd12m.csv'

def get_job_data(gpu_id, job_id, jobs_per_gpu=8, num_gpus=4):
    """Calculate the data chunk for this specific job"""
    # Read CSVs
    speakers_df = pd.read_csv(DEFAULT_SPEAKER_CSV)
    prompts_df = pd.read_csv(DEFAULT_PROMPT_CSV)
    
    total_jobs = jobs_per_gpu * num_gpus
    total_speakers = len(speakers_df)
    total_prompts = len(prompts_df)
    
    # Calculate speakers per job (rounded up to ensure all speakers are covered)
    speakers_per_job = -(total_speakers // -total_jobs)  # Ceiling division
    prompts_per_speaker = 10
    
    # Calculate start and end indices for speakers and prompts
    job_idx = (gpu_id - 1) * jobs_per_gpu + (job_id - 1)
    speaker_start_idx = job_idx * speakers_per_job
    speaker_end_idx = min(speaker_start_idx + speakers_per_job, total_speakers)
    
    prompt_start_idx = speaker_start_idx * prompts_per_speaker
    prompt_end_idx = speaker_end_idx * prompts_per_speaker
    
    # Get the relevant chunks of data
    job_speakers = speakers_df.iloc[speaker_start_idx:speaker_end_idx]
    job_prompts = prompts_df.iloc[prompt_start_idx:prompt_end_idx]
    
    return job_speakers, job_prompts

def main():
    parser = argparse.ArgumentParser(description="Batch XTTS Inference")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID (1-4)")
    parser.add_argument("--job_id", type=int, required=True, help="Job ID (1-8)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated audio")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set the GPU
    torch.cuda.set_device(args.gpu_id - 1)
    
    # Get the data for this job
    speakers_chunk, prompts_chunk = get_job_data(args.gpu_id, args.job_id)
    
    # Load the model
    model = load_model(DEFAULT_XTTS_CHECKPOINT, DEFAULT_XTTS_CONFIG, DEFAULT_XTTS_VOCAB)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each speaker
    prompts_per_speaker = 10
    for idx, speaker_row in tqdm(speakers_chunk.iterrows(), total=len(speakers_chunk)):
        speaker_id = speaker_row['speaker_id']
        speaker_audio = speaker_row['path']
        
        # Get the 10 prompts for this speaker
        start_prompt_idx = (idx - speakers_chunk.index[0]) * prompts_per_speaker
        speaker_prompts = prompts_chunk.iloc[start_prompt_idx:start_prompt_idx + prompts_per_speaker]
        
        # Generate audio for each prompt
        for prompt_idx, prompt_row in speaker_prompts.iterrows():
            output_filename = f"{speaker_id}_{prompt_idx}.wav"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Skip if already generated
            if os.path.exists(output_path):
                continue
                
            try:
                run_tts(
                    model=model,
                    lang="en",  # Fixed to English
                    tts_text=prompt_row['prompt'],
                    speaker_audio_file=speaker_audio,
                    output_path=output_path
                )
            except Exception as e:
                print(f"Error processing speaker {speaker_id}, prompt {prompt_idx}: {str(e)}")
                continue

if __name__ == "__main__":
    main() 