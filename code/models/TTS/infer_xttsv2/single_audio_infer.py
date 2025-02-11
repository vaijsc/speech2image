import argparse
import os
import tempfile
import time
import torch
import numpy as np
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    clear_gpu_cache()
    config = XttsConfig()
    config.load_json(xtts_config)
    xtts_model = Xtts.init_from_config(config)
    print("Loading XTTS model...")
    xtts_model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    if torch.cuda.is_available():
        xtts_model.cuda()
    print("Model loaded!")
    return xtts_model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_tts(model, lang, tts_text, speaker_audio_file, output_path):
    
    print(f"get_conditioning_latents params:")
    print(f"  audio_path={speaker_audio_file}")
    print(f"  gpt_cond_len={model.config.gpt_cond_len}")
    print(f"  max_ref_length={model.config.max_ref_len}")
    print(f"  sound_norm_refs={model.config.sound_norm_refs}")
    start_time = time.time()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=model.config.gpt_cond_len,         # 30
        max_ref_length=model.config.max_ref_len,        # 30
        sound_norm_refs=model.config.sound_norm_refs    # false
    )
    end_time = time.time()
    print(f'generating gpt cond latent takes {end_time - start_time} seconds')
    start_time = time.time()
    out = model.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=model.config.temperature,
        length_penalty=model.config.length_penalty,
        repetition_penalty=model.config.repetition_penalty,
        top_k=model.config.top_k,
        top_p=model.config.top_p,
        enable_text_splitting=True,
    )
    end_time = time.time()
    print(f'time taken to synthesize: {end_time - start_time} seconds')

    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    print(f'output length: {out["wav"].shape[1] / 24000}')
    torchaudio.save(output_path, out["wav"], 24000)
    print(f"Speech generated and saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTS Inference")
    parser.add_argument("--xtts_checkpoint", type=str, required=True, help="Path to the XTTS checkpoint")
    parser.add_argument("--xtts_config", type=str, required=True, help="Path to the XTTS config file")
    parser.add_argument("--xtts_vocab", type=str, required=True, help="Path to the XTTS vocab file")
    parser.add_argument("--lang", type=str, default="en", help="Language code for the TTS (default: 'en')")
    parser.add_argument("--text", type=str, required=True, help="Text to generate speech from")
    parser.add_argument("--speaker_audio", type=str, required=True, help="Path to the speaker reference audio")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to save the generated audio (default: 'output.wav')")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default: 42)")

    args = parser.parse_args()

    set_seed(args.seed)
    model = load_model(args.xtts_checkpoint, args.xtts_config, args.xtts_vocab)
    run_tts(model, args.lang, args.text, args.speaker_audio, args.output)