import time
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchaudio
import numpy as np
from TTS.tts.models.xtts import VoiceBpeTokenizer
import torch.multiprocessing as mp


class XTTSInferenceDataset(Dataset):
    """Dataset for batch inference with XTTS"""
    
    def __init__(
        self, 
        prompts_csv, 
        audio_csv, 
        prompt_dataset_name, 
        speech_dataset_name, 
        tokenizer, 
        sample_rate=24000, 
        max_text_length=250
    ):
        """
        Args:
            prompts_csv (str): Path to CSV containing text prompts
            audio_csv (str): Path to CSV containing speaker audio paths
            prompt_dataset_name (str): Name of the prompt dataset (e.g. 'pd12m')
            speech_dataset_name (str): Name of the speech dataset (e.g. 'commonvoice')
            tokenizer: XTTS tokenizer for text processing
            sample_rate (int): Audio sample rate
            max_text_length (int): Maximum text length in characters
        """
        
        # Basic setup
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_text_length = max_text_length
        self.prompt_dataset_name = prompt_dataset_name 
        self.speech_dataset_name = speech_dataset_name

        if not os.path.exists('samples.csv'):
            # Load CSVs
            print('loading csv...')
            start = time.time()
            self.prompts_df = pd.read_csv(prompts_csv)
            self.audio_df = pd.read_csv(audio_csv)
            print(f'done loading csv, it took: {time.time() - start} seconds')
        
            # Create prompt-speaker pairs
            # Each speaker gets 10 prompts
            speaker_ids = np.repeat(self.audio_df['speaker_id'].values, 10)
            if self.speech_dataset_name == 'commonvoice':
                speech_root = '/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/speech_prompts/0_common_voice/sampled_audio'
            paths = np.array([os.path.join(speech_root, path) for path in self.audio_df['path'].values])
            speaker_paths = np.repeat(paths, 10)
        
            # Take matching number of prompts
            prompts = self.prompts_df['prompt'].values[:len(speaker_ids)]
            prompt_ids = self.prompts_df['id'].values[:len(speaker_ids)]
        
            # Create final dataset
            self.samples = pd.DataFrame({
                'speaker_id': speaker_ids[:len(prompts)],
                'speaker_path': speaker_paths[:len(prompts)],
                'prompt_id': prompt_ids[:len(prompts)],
                'prompt': prompts
            })
            self.samples.to_csv('samples.csv', index=False)
        else: 
            print('samples.csv already exists. loading...')
            # TODO: change to full dataset
            self.samples = pd.read_csv('samples.csv').head(2)

    def get_text(self, text, lang='en'):
        """Convert text to token sequence using XTTS tokenizer.
        Follows the same logic as training dataset."""
        tokens = self.tokenizer.encode(text, lang)
        tokens = torch.IntTensor(tokens)
        # Verify no unknown tokens
        assert not torch.any(tokens == 1), f"UNK token found in {text} -> {self.tokenizer.decode(tokens)}"
        # The stop token should always be sacred
        assert not torch.any(tokens == 0), f"Stop token found in {text}"
        return tokens

    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a single sample for inference"""
        sample = self.samples.iloc[idx]
        
        # Get text tokens
        text = str(sample['prompt'])
        # Truncate text if needed
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        tokens = self.get_text(text)
        
        return {
            'text': tokens,
            'text_lengths': torch.tensor(tokens.shape[0], dtype=torch.long),
            'audio_path': sample['speaker_path'],  # Return path instead of waveform
            'speaker_id': sample['speaker_id'],
            'prompt': text,
            'prompt_id': sample['prompt_id']
        }

    def collate_fn(self, batch):
        """Collate batch of samples"""
        # Get max lengths
        max_text_len = max(x['text_lengths'] for x in batch)
        
        # Initialize tensors
        B = len(batch)
        text_padded = torch.zeros((B, max_text_len), dtype=torch.long)
        
        # Fill in the tensors
        text_lengths = []
        audio_paths = []  # Store paths instead of waveforms
        speaker_ids = []
        prompts = []
        prompt_ids = []
        
        for i, b in enumerate(batch):
            text = b['text']
            text_padded[i, :len(text)] = text
            text_lengths.append(b['text_lengths'])
            
            audio_paths.append(b['audio_path'])
            speaker_ids.append(b['speaker_id'])
            prompts.append(b['prompt'])
            prompt_ids.append(b['prompt_id'])
            
        return {
            'text': text_padded,
            'text_lengths': torch.stack(text_lengths),
            'audio_paths': audio_paths,  # Return paths
            'speaker_ids': speaker_ids,
            'prompts': prompts,
            'prompt_ids': prompt_ids
        }


if __name__ == '__main__':
    def load_model():
        print(" > Loading XTTS model and configs...")
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        import torch

        config = XttsConfig()
        config.load_json("checkpoints/config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, 
                              checkpoint_dir="checkpoints",
                              use_deepspeed=False
                              )
        model.to("cuda")
        return model

    # Load the model
    model = load_model()

    def _test_dataset():
        # Test paths
        prompts_csv = "/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/text_prompts/shorten_csvs/pd12m.csv"
        audio_csv = "/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/speech_prompts/0_common_voice/sampled_audio.csv"

        # Initialize tokenizer
        vocab_path = "checkpoints/vocab.json"
        tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        # Create dataset
        print(" > Creating dataset...")
        dataset = XTTSInferenceDataset(
            prompts_csv=prompts_csv,
            audio_csv=audio_csv,
            prompt_dataset_name='pd12m', 
            speech_dataset_name='commonvoice',
            tokenizer=tokenizer
        )
        
        # Test single item loading
        print("\n> Testing single item loading...")
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Text shape: {sample['text'].shape}")
        print(f"Audio path: {sample['audio_path']}")  # Print audio path instead of waveform shape
        print(f"Text length: {sample['text_lengths']}")
        print(f"First few tokens: {sample['text'][:10]}")
        print(f"Original prompt: {sample['prompt'][:100]}...")
        print(f"Decoded tokens: {tokenizer.decode(sample['text'].tolist())}")

    def _test_dataloader():
        # Test paths
        prompts_csv = "/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/text_prompts/shorten_csvs/pd12m.csv"
        audio_csv = "/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/speech_prompts/0_common_voice/sampled_audio.csv"

        # Initialize tokenizer
        vocab_path = "checkpoints/vocab.json"
        tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        # Create dataset
        dataset = XTTSInferenceDataset(
            prompts_csv=prompts_csv,
            audio_csv=audio_csv,
            prompt_dataset_name='pd12m',
            speech_dataset_name='commonvoice',
            tokenizer=tokenizer
        )
        
        # Test dataloader
        print("\n> Testing dataloader...")
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Keep 0 for testing
            collate_fn=dataset.collate_fn
        )
        
        # Get first batch
        print("Getting first batch...")
        batch = next(iter(dataloader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch text shape: {batch['text'].shape}")
        print(f"Number of audio paths: {len(batch['audio_paths'])}")  # Print number of paths
        print(f"First audio path: {batch['audio_paths'][0]}")  # Print first path
        print(f"Batch text lengths: {batch['text_lengths']}")
        
        # Test a few batches
        print("\n> Testing multiple batches...")
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Just test first few batches
                break
            print(f"\nBatch {i}:")
            print(f"Text shape: {batch['text'].shape}")
            print(f"Number of audio paths: {len(batch['audio_paths'])}")
            print(f"Number of samples: {len(batch['speaker_ids'])}")
            print(f"First prompt: {batch['prompts'][0][:50]}...")
            print(f"First audio path: {batch['audio_paths'][0]}")
            print(f"Decoded tokens: {tokenizer.decode(batch['text'][0].tolist())}")

    # Run tests
    print("=== Testing Dataset ===")
    _test_dataset()
    
    print("\n=== Testing Dataloader ===")
    _test_dataloader() 