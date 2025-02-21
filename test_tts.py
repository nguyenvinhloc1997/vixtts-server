#!/usr/bin/env python3
import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ============ EDIT THESE PARAMETERS ============
# Text to convert to speech
TEXT = "Em chào anh chị, anh chị có nhu cầu chuyển nhượng hay cho thuê căn hộ chung cư ở Golden Palace không ạ?"

# Language code (vi, en, fr, etc.)
LANGUAGE = "vi"

# Path to reference speaker WAV file
SPEAKER_WAV = "assets/hong.wav"

# Where to save the output
OUTPUT_FILE = "output/hong_speech.wav"

# Model settings
MODEL_PATH = "model/"
USE_DEEPSPEED = True

# Generation parameters
TEMPERATURE = 0.3        # Controls randomness (0.1-1.0)
LENGTH_PENALTY = 1.0     # Controls output length
REPETITION_PENALTY = 10.0  # Prevents repetition
TOP_K = 30              # Top-k sampling parameter
TOP_P = 0.85            # Top-p sampling parameter
SPEED = 1.2             # Speech speed (0.5 for slower, 2.0 for faster)
# ============================================

def load_model(model_path="model/", use_deepspeed=True):
    """Load the TTS model"""
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=use_deepspeed)
    if torch.cuda.is_available():
        model.cuda()
    return model

def generate_speech(
    model,
    text,
    language="vi",
    speaker_wav=None,
    output_file=None,
    temperature=0.3,
    length_penalty=1.0,
    repetition_penalty=10.0,
    top_k=30,
    top_p=0.85,
    speed=1.0
):
    """Generate speech from text"""
    # Get speaker conditioning
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_wav,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )

    # Generate speech
    out = model.inference(
        text=text,
        language=language,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        enable_text_splitting=True,
    )

    # Apply speed adjustment if needed
    if speed != 1.0:
        wav = torch.tensor(out['wav'])
        if speed > 0:
            # Resample to adjust speed
            old_sample_rate = 24000
            new_sample_rate = int(old_sample_rate * speed)
            wav = torchaudio.transforms.Resample(
                old_sample_rate, new_sample_rate
            )(wav.unsqueeze(0))
            wav = torchaudio.transforms.Resample(
                new_sample_rate, old_sample_rate
            )(wav).squeeze(0)
            out['wav'] = wav.numpy()

    # Save the output
    if output_file:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        wav = torch.tensor(out['wav']).unsqueeze(0)
        torchaudio.save(output_file, wav, 24000)
    
    return out['wav']

def main():
    try:
        print("Loading model...")
        model = load_model(MODEL_PATH, use_deepspeed=USE_DEEPSPEED)
        
        print("Generating speech...")
        print(f"Text: {TEXT}")
        print(f"Language: {LANGUAGE}")
        print(f"Using speaker reference: {SPEAKER_WAV}")
        
        generate_speech(
            model=model,
            text=TEXT,
            language=LANGUAGE,
            speaker_wav=SPEAKER_WAV,
            output_file=OUTPUT_FILE,
            temperature=TEMPERATURE,
            length_penalty=LENGTH_PENALTY,
            repetition_penalty=REPETITION_PENALTY,
            top_k=TOP_K,
            top_p=TOP_P,
            speed=SPEED
        )
        print(f"Speech generated and saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 