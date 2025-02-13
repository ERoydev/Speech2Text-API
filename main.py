from fastapi import FastAPI, UploadFile, File
from utils import load_audio_file
from faster_whisper import WhisperModel
import tempfile
import os
app = FastAPI()

# Load Faster Whisper model (force CPU mode to avoid CUDA errors)
# This is the smallest possible model & optimize memory
model = WhisperModel("tiny", device="cpu", compute_type="int8")  # Use "cpu" to avoid CUDA errors

@app.post("/audio_transcription")
async def audio_transcribe(file: UploadFile = File(...)): # This requres multipart/formdata
    file_bytes = await file.read()
    
    audio = load_audio_file(file_bytes)
    audio_duration = len(audio) / 1000.0 # Audio Duration in seconds
    
    """Handles saving, transcribing, and cleaning up the audio file using Faster Whisper."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio.export(temp_audio_file.name, format="wav")
        temp_audio_file.flush()  # Ensure all data is written
    
    # Close the file to release it before processing
    temp_audio_file_path = temp_audio_file.name
    os.chmod(temp_audio_file_path, 0o644)  # Set read permissions
    
    try:
        # Transcribe using Faster Whisper
        segments, info = model.transcribe(temp_audio_file_path)
    
        # Extract transcribed text
        transcription_text = " ".join([segment.text for segment in segments])
    
        # Reformat result to match old Whisper output
        result = {
            "text": transcription_text,
            "segments": [  # Add segment details like Whisper does
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                for segment in segments
            ],
            "language": info.language
        }
    
        return {
            "transcription_text": transcription_text,
            "transcribed_audio": result,
            "audio_duration": audio_duration,
        }
    finally:
        # Cleanup the temporary file after transcription
        os.remove(temp_audio_file_path)
