import numpy as np
import time
import torch
import torchaudio
from torch.utils.data import DataLoader
from pathlib import Path
from TTS.tts.models.xtts import VoiceBpeTokenizer
from inference_dataset import XTTSInferenceDataset


def run_batch_inference(
    model,
    prompts_csv,
    audio_csv,
    output_dir,
    batch_size=8,
    num_workers=4,
    device="cuda",
    sample_rate=24000
):
    """
    Run batch inference with XTTS
    
    Args:
        model: XTTS model
        prompts_csv: Path to prompts CSV
        audio_csv: Path to speaker audio CSV  
        output_dir: Directory to save generated audio
        batch_size: Batch size for inference
        num_workers: Number of dataloader workers
        device: Device to run inference on
        sample_rate: Audio sample rate for saving files
    """
    # Initialize tokenizer
    vocab_path = "checkpoints/vocab.json"
    tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
    
    # Create dataset and dataloader
    dataset = XTTSInferenceDataset(
        prompts_csv=prompts_csv,
        audio_csv=audio_csv,
        prompt_dataset_name='pd12m',
        speech_dataset_name='commonvoice',
        tokenizer=tokenizer
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx}")
            
            # Get conditioning latents and speaker embeddings for the batch
            gpt_cond_latents = []
            speaker_embeddings = []
            
            start = time.time()
            for audio_path in batch['audio_paths']:
                # Get conditioning latents for each audio in batch
                latents, embedding = model.get_conditioning_latents(
                    audio_path=audio_path,
                    max_ref_length=30,
                    gpt_cond_len=30,
                    gpt_cond_chunk_len=6,
                    sound_norm_refs=False
                )
                gpt_cond_latents.append(latents)
                speaker_embeddings.append(embedding)
            end = time.time()
            print(f'time taken to calculate cond latent: {end - start} seconds')
            
            # Stack batch tensors
            gpt_cond_latents = torch.cat(gpt_cond_latents, dim=0)
            speaker_embeddings = torch.cat(speaker_embeddings, dim=0)
            
            # Move text inputs to device
            text = batch['text'].to(device)
            # text_lengths = batch['text_lengths'].to(device)
            
            # Generate audio
            start = time.time()
            outputs = model.batch_inference(
                texts=batch['prompts'],
                language='en',
                gpt_cond_latents=gpt_cond_latents,
                speaker_embeddings=speaker_embeddings
            )
            end = time.time()
            print(f'time take to synthesize audio: {end - start} seconds')
            
            breakpoint()
            # Save outputs
            for i, (speaker_id, prompt_id, prompt) in enumerate(zip(batch['speaker_ids'], batch['prompt_ids'], batch['prompts'])):
                output_path = output_dir / f"{speaker_id}_{prompt_id}.wav"
                torchaudio.save(output_path, outputs['wav'][i], sample_rate)
                print(f"prompt: {prompt}, \nsave path: {output_path}")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    import torch

    set_seed(42)
    def test_batch_inference():
        print(" > Loading XTTS model and configs...")
        config = XttsConfig()
        config.load_json("checkpoints/config.json")
        
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, 
                              checkpoint_dir="checkpoints",
                              use_deepspeed=False
                              )
        model.to("cuda")
        
        # Test paths
        prompts_csv = "/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/text_prompts/shorten_csvs/pd12m.csv"
        audio_csv = "/lustre/scratch/client/vinai/users/thivt1/speech2image/data/processed/speech_prompts/0_common_voice/sampled_audio.csv"
        output_dir = "test_outputs"
        
        # Run batch inference with small batch for testing
        print("\n > Running batch inference...")
        run_batch_inference(
            model=model,
            prompts_csv=prompts_csv,
            audio_csv=audio_csv,
            output_dir=output_dir,
            batch_size=2,  # Small batch for testing
            num_workers=2
        )
        
        # Verify outputs
        # output_dir = Path(output_dir)
        # generated_files = list(output_dir.glob("*.wav"))
        # print(f"\n > Generated {len(generated_files)} audio files")
        # print(" > First few files:")
        # for file in generated_files[:3]:
        #     print(f"   - {file.name}")

    # Run test
    print("=== Testing Batch Inference ===")
    test_batch_inference() 