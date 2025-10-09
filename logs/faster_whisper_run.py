
from faster_whisper import WhisperModel
import os

def format_timestamp(seconds: float):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) % 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def generate_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i+1}\n{start} --> {end}\n{text}\n\n")

def main():
    video_path = "/path/to/your/video.mp4"
    srt_path = "./output.srt"
    model_size = "large-v3"  # or "base", "medium", etc.
    language = "ja"  # Japanese
    device = "cuda:0"  # You can change to cuda:1, cuda:2, etc. for multi-GPU
    compute_type = "float16"  # or "int8_float16" for speed

    # Load model
    print(f"Loading Faster-Whisper model ({model_size}) on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Transcribe
    print("Transcribing...")
    segments, info = model.transcribe(video_path, language=language, beam_size=5)

    print("Saving SRT...")
    generate_srt(segments, srt_path)
    print(f"✅ Subtitle saved to {srt_path}")

if __name__ == "__main__":
    main()
