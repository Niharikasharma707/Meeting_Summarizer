import os
from pyannote.audio import Pipeline
import speech_recognition as sr
from transformers import pipeline as transformers_pipeline
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException
from typing import Optional
import tempfile
import shutil

# Initialize FastAPI app
app = FastAPI()

# Set your Hugging Face token
HUGGING_FACE_TOKEN = "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"

# Initialize the Pyannote speaker diarization pipeline
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGING_FACE_TOKEN
    )
    print("Diarization pipeline initialized successfully.")
except Exception as e:
    print(f"Error initializing diarization pipeline: {e}")
    diarization_pipeline = None

# Initialize the summarizer pipeline

# Initialize the summarizer pipeline
try:
    summarizer = transformers_pipeline("summarization")
    print("Summarizer initialized successfully.")
except Exception as e:
    print(f"Error initializing summarizer: {e}")
    summarizer = None


def convert_audio_to_pcm_wav(audio_file: str) -> Optional[str]:
    """Convert audio file to PCM WAV format if it's not already in that format."""
    if audio_file.endswith(".wav"):
        return audio_file  # No conversion needed

    try:
        audio = AudioSegment.from_file(audio_file)
        pcm_wav_file = audio_file.rsplit('.', 1)[0] + "_converted.wav"
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(pcm_wav_file, format="wav")
        print(f"Audio file converted to PCM WAV: {pcm_wav_file}")
        return pcm_wav_file
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        return None


def perform_speaker_diarization(audio_file: str):
    """Perform speaker diarization on the given audio file."""
    if not diarization_pipeline:
        print("Error: Diarization pipeline is not initialized.")
        return []

    print("Performing speaker diarization...")
    try:
        diarization_result = diarization_pipeline(audio_file)
        diarization_segments = []

        print("Diarization result:")
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            diarization_segments.append({"speaker": speaker, "start": turn.start, "end": turn.end})
            print(f"Speaker {speaker}: {turn.start:.1f}s to {turn.end:.1f}s")

        return diarization_segments
    except Exception as e:
        print(f"Error during speaker diarization: {e}")
        return []


def transcribe_audio(audio_file: str, diarization_segments):
    """Transcribe the audio file and align transcription with speaker segments."""
    print("Transcribing audio...")
    recognizer = sr.Recognizer()
    speaker_transcriptions = {}

    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

        # Group transcriptions by speaker
        for segment in diarization_segments:
            speaker = segment["speaker"]
            start, end = segment["start"], segment["end"]
            # You could use the start and end times for better segmentation, if needed
            if speaker not in speaker_transcriptions:
                speaker_transcriptions[speaker] = []
            speaker_transcriptions[speaker].append(transcription)  # Simplified, could split by time

        print("Transcription completed.")
        for speaker, transcriptions in speaker_transcriptions.items():
            print(f"Speaker {speaker}: {' '.join(transcriptions)}")

        return speaker_transcriptions
    except Exception as e:
        print(f"Error during transcription: {e}")
        return {}


def summarize_text(text: str):
    """Summarize the transcribed text."""
    if not summarizer:
        print("Error: Summarizer is not initialized.")
        return "Summarizer is not available."

    print("Summarizing transcription...")
    try:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error during summarization."


def analyze_audio(audio_file: str):
    """Analyze the audio file and summarize the content."""
    # Check if the audio file exists
    if not os.path.exists(audio_file):
        print("Error: Audio file does not exist.")
        return None

    audio_file = convert_audio_to_pcm_wav(audio_file)
    if not audio_file:
        print("Error: Failed to convert audio to PCM WAV.")
        return None

    diarization_segments = perform_speaker_diarization(audio_file)
    speaker_transcriptions = transcribe_audio(audio_file, diarization_segments)

    # Combine all text for summarization
    full_text = " ".join([st for transcriptions in speaker_transcriptions.values() for st in transcriptions])

    if full_text:
        summary = summarize_text(full_text)
        print("\nSummary:")
        print(summary)
    else:
        summary = "Error during transcription, no summary available."

    return {"speaker_transcriptions": speaker_transcriptions, "summary": summary}


# Define GET endpoint for analyzing audio
@app.get("/analyze-audio")
async def analyze_audio_endpoint(audio_file: str):
    """API endpoint to analyze audio file."""
    try:
        # Check if the audio file exists
        if not os.path.exists(audio_file):
            raise HTTPException(status_code=404, detail="Audio file not found")

        # Process the audio
        result = analyze_audio(audio_file)

        if result:
            return {"speaker_transcriptions": result['speaker_transcriptions'], "summary": result['summary']}
        else:
            raise HTTPException(status_code=500, detail="Error processing audio")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI server
# Use 'uvicorn filename:app --reload' to run the server
