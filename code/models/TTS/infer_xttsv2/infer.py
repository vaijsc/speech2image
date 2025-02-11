import re
import numpy as np
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import pandas as pd
import argparse
from tqdm import tqdm


'''
1. load csv data
2. get the subset of samples based on gpu_id and job_id
3. run inference on those samples 
4. save wav file in the followin format: {speaker_id}_{prompt_id}.wav
'''

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_job_samples(gpu_id, job_id): 
    # Calculate the total number of jobs and the number of jobs per GPU
    jobs_per_gpu = 8
    num_gpus = 6
    total_jobs = jobs_per_gpu * num_gpus

    # Calculate the data chunk for this specific job
    total_samples = len(samples_df)
    samples_per_job = -(total_samples // -total_jobs)  # Ceiling division, hence the -

    # Calculate start and end indices for the current job
    job_idx = (gpu_id - 1) * jobs_per_gpu + (job_id - 1)
    start_idx = job_idx * samples_per_job
    end_idx = min(start_idx + samples_per_job, total_samples)
    print(f' > this job will run on samples from index {start_idx} to {end_idx}')

    # Get the relevant chunk of data for this job
    job_samples = samples_df.iloc[start_idx:end_idx]
    return job_samples


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model():
    xtts_checkpoint = 'checkpoints/model.pth'
    xtts_config = 'checkpoints/config.json'
    xtts_vocab = 'checkpoints/vocab.json'
    clear_gpu_cache()
    config = XttsConfig()
    config.load_json(xtts_config)
    xtts_model = Xtts.init_from_config(config)
    xtts_model.load_checkpoint(config, 
                               checkpoint_path=xtts_checkpoint, 
                               vocab_path=xtts_vocab, 
                               use_deepspeed=False)
    if torch.cuda.is_available():
        xtts_model.cuda().eval()
    return xtts_model


def speaker_info_saved(speaker_id):
    if os.path.exists(f'./speaker_emds/{speaker_id}.pth'):
        return True
    return False


def run_tts(model, tts_text, speaker_audio_file, speaker_id, output_path):
    # encode speaker info
    if speaker_info_saved(speaker_id):
        gpt_cond_latent = torch.load(f'./speaker_cond_latents/{speaker_id}.pth')
        speaker_embd = torch.load(f'./speaker_embs/{speaker_id}.pth')
    else:
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=model.config.gpt_cond_len,         # 30
            max_ref_length=model.config.max_ref_len,        # 30
            sound_norm_refs=model.config.sound_norm_refs    # false
        )
        # save speaker info
        torch.save(gpt_cond_latent, f'./speaker_cond_latents/{speaker_id}.pth')
        torch.save(speaker_embedding, f'./speaker_embs/{speaker_id}.pth')
    
    # gpt generate audio
    out = model.inference(
        text=tts_text,
        language='en',
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=model.config.temperature,
        length_penalty=model.config.length_penalty,
        repetition_penalty=model.config.repetition_penalty,
        top_k=model.config.top_k,
        top_p=model.config.top_p,
        enable_text_splitting=True,
    )

    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    torchaudio.save(output_path, out["wav"], 24000)


def count_num(text): 
    '''
    count the number of numbers in a text string. example:
    'The image shows a large grassy field with trees in the background, located at Lot 1, Lot 2, Lot 3,...'
    '''
    numbers = re.findall(r'\b\d+\b', text) # list of numbers: ['1', '2', '3']
    return len(numbers)


if __name__ == '__main__': 
    # 0. first thing first, let's set seed
    set_seed(1)

    # 1. load csv data
    print(" > Loading samples from 'samples.csv'...")
    samples_df = pd.read_csv('samples.csv') # ['speaker_id', 'speaker_path', 'prompt_id', 'prompt']

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process GPU ID and Job ID")
    parser.add_argument("--gpu_id", type=int, help="GPU ID (1-6)")
    parser.add_argument("--job_id", type=int, help="Job ID (1-8)")
    args = parser.parse_args()

    gpu_id = args.gpu_id
    job_id = args.job_id

    # 2. get the subset of samples based on gpu_id and job_id
    print(' > Getting the subset of samples for this job')
    job_samples = get_job_samples(gpu_id, job_id)

    # 3. run inference on those samples 
    print(" > Loading XTTS model and configs...")
    model = load_model()
    root_save_dir = '/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/generated_speech'
    speech_dataset = "cv"
    text_dataset = "pd12m"
    print(' > Doing inference...')
    for index, sample in tqdm(job_samples.iterrows(), total=job_samples.shape[0], desc="infering samples"):
        output_path = os.path.join(root_save_dir, 
                                   f"{speech_dataset}_{sample['speaker_id']}-{text_dataset}_{sample['prompt_id']}.wav")
        text = sample['prompt']
        if not os.path.exists(output_path) and count_num(text) < 3:
            run_tts(model, 
                    tts_text=text, 
                    speaker_audio_file=sample['speaker_path'], 
                    speaker_id=sample['speaker_id'],
                    output_path=output_path)
