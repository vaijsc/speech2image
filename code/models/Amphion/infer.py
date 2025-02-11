from models.tts.maskgct.maskgct_utils import *
from huggingface_hub import hf_hub_download
import safetensors
import soundfile as sf


def init_pipeline():
    # download semantic codec ckpt
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")

    # download acoustic codec ckpt
    codec_encoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model.safetensors")
    codec_decoder_ckpt = hf_hub_download("amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors")

    # download t2s model ckpt
    t2s_model_ckpt = hf_hub_download("amphion/MaskGCT", filename="t2s_model/model.safetensors")

    # download s2a model ckpt
    s2a_1layer_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_1layer/model.safetensors")
    s2a_full_ckpt = hf_hub_download("amphion/MaskGCT", filename="s2a_model/s2a_model_full/model.safetensors")

    # build model
    device = torch.device("cuda:0")
    cfg_path = "./models/tts/maskgct/config/maskgct.json"
    cfg = load_config(cfg_path)
    # 1. build semantic model (w2v-bert-2.0)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    # 2. build semantic codec
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    # 3. build acoustic codec
    codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
    # 4. build t2s model
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    # 5. build s2a model
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full =  build_s2a_model(cfg.model.s2a_model.s2a_full, device)


    # load semantic codec
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    # load acoustic codec
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    # load t2s model
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    # load s2a model
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )
    return maskgct_inference_pipeline

if __name__ == "__main__":
    maskgct_inference_pipeline = init_pipeline()

    # inference
    prompt_wav_path = "/home/dungnt206/common_voice_en_18365693.mp3"
    save_path = "./demo.wav"
    prompt_text = "You are so rude."
    target_text = "The image shows a painting of a woman in a white dress, framed in a photo frame. Her face is clearly visible, with her eyes looking off to the side and her lips slightly parted. Her hair is pulled back and she has a gentle expression on her face."
    # Specify the target duration (in seconds). If target_len = None, we use a simple rule to predict the target duration.
    target_len = None 

    recovered_audio = maskgct_inference_pipeline.maskgct_inference(
        prompt_wav_path, prompt_text, target_text, "en", "en", target_len=target_len
    )
    sf.write(save_path, recovered_audio, 24000)
