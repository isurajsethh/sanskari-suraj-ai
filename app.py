import os, tempfile, subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)  # For demo use. Restrict origins to your app domain for production.
app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024

@app.get('/')
def home():
    return jsonify({'service':'Sanskari Suraj Lyrics Typer API','status':'online','endpoint':'POST /transcribe'})

@app.post('/transcribe')
def transcribe():
    if not os.getenv('GROQ_API_KEY'):
        return jsonify({'error':'Server admin has not configured GROQ_API_KEY'}), 500
    uploaded = request.files.get('file')
    if not uploaded or not uploaded.filename:
        return jsonify({'error':'No video/audio file uploaded in field named file'}), 400
    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, 'input_upload')
        audio_path = os.path.join(tmp, 'audio.mp3')
        uploaded.save(input_path)
        # Extract mono audio and compress to keep within API upload limit.
        cmd = ['ffmpeg','-y','-i',input_path,'-vn','-ac','1','-ar','16000','-b:a','48k',audio_path]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if run.returncode != 0 or not os.path.exists(audio_path):
            return jsonify({'error':'Video/audio could not be decoded. Try MP4, MP3, M4A or WAV.'}), 400
        if os.path.getsize(audio_path) > 24 * 1024 * 1024:
            return jsonify({'error':'Audio is still over 24 MB. Try a shorter video.'}), 413
        try:
            client = Groq(api_key=os.environ['GROQ_API_KEY'])
            with open(audio_path, 'rb') as f:
                result = client.audio.transcriptions.create(
                    file=('audio.mp3', f.read()),
                    model='whisper-large-v3-turbo',
                    response_format='json',
                    temperature=0.0
                )
            return jsonify({'text': result.text})
        except Exception as e:
            # Do not expose secret keys or full server internals.
            return jsonify({'error':'AI transcription request failed. Check API key, usage limits and server logs.', 'detail':str(e)[:240]}), 502

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8080')))
