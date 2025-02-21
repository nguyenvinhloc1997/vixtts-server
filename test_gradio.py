#!/usr/bin/env python3
from gradio_client import Client, handle_file

# Initialize the client
client = Client("http://localhost:5004")

def load_model(checkpoint_dir="model", repo_id="capleaf/viXTTS", use_deepspeed=True):
    """Load the TTS model"""
    return client.predict(
        checkpoint_dir,
        repo_id,
        use_deepspeed,
        api_name="/load_model"
    )

def run_tts(text, language="vi", speaker_wav_path=None, use_deepfilter=True, normalize_text=True):
    """Generate speech using the TTS model"""
    # If speaker_wav_path is a local file, we need to handle it properly
    speaker_audio = handle_file(speaker_wav_path) if speaker_wav_path else None
    
    return client.predict(
        language,              # Language code
        text,                  # Text to convert to speech
        speaker_audio,         # Speaker reference audio
        use_deepfilter,       # Whether to denoise reference audio
        normalize_text,        # Whether to normalize input text
        api_name="/run_tts"
    )

def main():
    try:
        # Parameters - feel free to modify these
        TEXT = "Em chào anh chị, anh chị có nhu cầu chuyển nhượng hay cho thuê căn hộ chung cư ở Golden Palace không ạ?"
        LANGUAGE = "vi"
        SPEAKER_WAV = "assets/hong.wav"  # Using the default sample audio
        USE_DEEPFILTER = True
        NORMALIZE_TEXT = True
        
        print("Loading model...")
        result = load_model()
        print("Model loading result:", result)
        
        print("\nGenerating speech...")
        print(f"Text: {TEXT}")
        print(f"Language: {LANGUAGE}")
        print(f"Using speaker reference: {SPEAKER_WAV}")
        
        progress, audio_path = run_tts(
            text=TEXT,
            language=LANGUAGE,
            speaker_wav_path=SPEAKER_WAV,
            use_deepfilter=USE_DEEPFILTER,
            normalize_text=NORMALIZE_TEXT
        )
        
        print("Generation progress:", progress)
        print("Generated audio saved to:", audio_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    main() 