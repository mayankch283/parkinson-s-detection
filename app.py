from flask import Flask, request, render_template, jsonify
from fastai.learner import load_learner
from pathlib import Path
import librosa
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the exported model
model = load_learner('predictor.pkl')

def get_spectrogram(audio_file):
    y, sr = librosa.load(audio_file, duration=5)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def spec_to_image(spec):
    spec_normalized = (spec - spec.min()) / (spec.max() - spec.min())
    spec_rgb = np.repeat(spec_normalized[..., np.newaxis], 3, axis=-1)
    img = Image.fromarray(np.uint8(spec_rgb * 255))
    return img

def get_spec_image(audio_file):
    spec = get_spectrogram(audio_file)
    return spec_to_image(spec)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            # Process the audio file
            img = get_spec_image(file)
            
            # Make prediction
            pred, _, probs = model.predict(img)
            
            return jsonify({
                'prediction': str(pred),
                'probabilities': {
                    'healthy': float(probs[0]),
                    'parkinsons': float(probs[1])
                }
            })
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)