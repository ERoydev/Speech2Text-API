from pydub import AudioSegment
import io

def load_audio_file(audio_bytes: bytes):
    """
    Load audio file from bytes using pydub.
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        return audio
    except Exception as e:
        raise ValueError(f"Error loading audio file: {str(e)}")
