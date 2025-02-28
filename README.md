# Speech2Text-API

WhisperTranscriber is a lightweight microservice that leverages [faster_whisper](https://github.com/guillaumekln/faster-whisper) to convert speech to text. Built with FastAPI and pydub, it provides a simple REST API endpoint for uploading audio files and receiving transcriptions.

## Features

- **Speech-to-Text Transcription:** Uses faster_whisper to transcribe `.wav` audio files.
- **REST API:** A POST endpoint to upload audio files and get transcriptions.
- **Temporary File Handling:** Uses Python's `tempfile` to manage audio files during processing.
- **Asynchronous Processing:** Built on FastAPI to handle concurrent requests.

## Requirements

- Python 3.8+
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [faster_whisper](https://github.com/guillaumekln/faster-whisper)
- [pydub](https://github.com/jiaaro/pydub)
- [ffmpeg](https://ffmpeg.org/) (required by pydub)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/whisper-transcriber.git
   cd whisper-transcriber


2. **Examples:**

- Create a POST request using whatever http client you want and make a request to this url /audio_transcription
- You need to have audio.wav file and create multipart/formdata
```py
    with requests.Session() as session: # THIS IS REQUEST TO MY MICROSERVICE TO GET THE TRANSCIPTION => TODO: IMPLEMENT ASYNCHRONOUS BEHAVIOUR
        files = {'file': (audio_file.name, audio_file, audio_file.content_type)}  # I create multipart/formdata
        response = session.post('http://127.0.0.1:9000/audio_transcription', files=files)

  You need to use ur server url and post request to /audio_transcription
```

The microservice response is:
```py
{
    "transcription_text": " Hello, I want to test the functionality of my backend. I don't know why I am getting these errors.",
    "transcribed_audio": {
        "text": " Hello, I want to test the functionality of my backend. I don't know why I am getting these errors.",
        "segments": [],
        "language": "en"
    },
    "audio_duration": 7.32
}
```
