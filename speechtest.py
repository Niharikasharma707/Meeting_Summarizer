import os
from pyannote.audio import Pipeline
import speech_recognition as sr
from transformers import pipeline as transformers_pipeline
from pydub import AudioSegment
import tempfile
import wave
import webrtcvad  # Voice Activity Detection
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
try:
    summarizer = transformers_pipeline("summarization")
    print("Summarizer initialized successfully.")
except Exception as e:
    print(f"Error initializing summarizer: {e}")
    summarizer = None
# Initialize VAD (WebRTC Voice Activity Detection)
vad = webrtcvad.Vad(3)  # 3 for aggressive VAD (you can change this level)
def convert_audio_to_pcm_wav(audio_file):
    """Convert audio file to PCM WAV format if it's not already in that format."""
    if audio_file.endswith(".wav"):
        return audio_file  # No conversion needed
    try:
        audio = AudioSegment.from_file(audio_file)
        pcm_wav_file = audio_file.rsplit('.', 1)[0] + "_converted.wav"
        audio = audio.set_channels(1).set_frame_rate(16000)  # Convert to mono and 16kHz PCM WAV
        audio.export(pcm_wav_file, format="wav")
        print(f"Audio file converted to PCM WAV: {pcm_wav_file}")
        return pcm_wav_file
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        return None
def apply_vad(audio_file):
    """Apply Voice Activity Detection (VAD) to filter out silence."""
    print("Applying VAD to audio...")
    with wave.open(audio_file, 'rb') as f:
        framerate = f.getframerate()
        frames = f.readframes(f.getnframes())
    # Split audio into chunks and apply VAD
    chunks = []
    for i in range(0, len(frames), framerate):
        chunk = frames[i:i+framerate]
        if vad.is_speech(chunk, framerate):
            chunks.append(chunk)
    return chunks
def perform_speaker_diarization(audio_file):
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
def transcribe_audio(audio_file, diarization_segments):
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
def summarize_text(text):
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
def analyze_audio(audio_file):
    """Analyze the audio file and summarize the content."""
    # Check if the audio file exists
    if not os.path.exists(audio_file):
        print("Error: Audio file does not exist.")
        return None
    audio_file = convert_audio_to_pcm_wav(audio_file)
    if not audio_file:
        print("Error: Failed to convert audio to PCM WAV.")
        return None
    # Apply VAD to filter out silent parts
    audio_chunks = apply_vad(audio_file)
    # Save the VAD-filtered audio back as a new file (you can modify this to handle real-time streams)
    temp_audio_file = tempfile.mktemp(suffix=".wav")
    with wave.open(temp_audio_file, 'wb') as out_f:
        out_f.setnchannels(1)
        out_f.setsampwidth(2)
        out_f.setframerate(16000)
        out_f.writeframes(b''.join(audio_chunks))
    # Perform speaker diarization on the VAD-filtered audio
    diarization_segments = perform_speaker_diarization(temp_audio_file)
    # Transcribe and align transcription with diarization
    speaker_transcriptions = transcribe_audio(temp_audio_file, diarization_segments)
    # Combine all text for summarization
    full_text = " ".join([st for transcriptions in speaker_transcriptions.values() for st in transcriptions])
    if full_text:
        summary = summarize_text(full_text)
        print("\nSummary:")
        print(summary)
    else:
        summary = "Error during transcription, no summary available."
    return {"speaker_transcriptions": speaker_transcriptions, "summary": summary}
# Example usage
if __name__ == "__main__":
    # Replace with the path to your audio file
    audio_file = "C:/Users/Signity_Laptop/notemaker/Catching Up With Friends Audio 3.wav"
    result = analyze_audio(audio_file)
    if result:
        print("\nFinal Result:")
        print("Speaker Transcriptions:")
        for speaker, transcriptions in result['speaker_transcriptions'].items():
            print(f"Speaker {speaker}: {' '.join(transcriptions)}")
        print("\nSummary:")
        print(result['summary']) 