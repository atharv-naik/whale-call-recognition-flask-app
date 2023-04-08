import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask, render_template, request
import numpy as np
from preprocess import extract_featuresH, extract_featuresV
from make_aiff import convert_to_aiff
from keras.models import load_model
import warnings
# Suppress warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav', 'aiff', 'mp3'}
MAX_FILESIZE_BYTES = 5000 * 1024 # 5 MB

def get_extension(filename):
    if '.' not in filename:
        return None
    else:
        return filename.rsplit('.', 1)[1].lower()

def preprocess(audio_path):
    # Extract the features
    params = {'NFFT': 256, 'Fs': 2000, 'noverlap': 192}
    X_testV = np.array([extract_featuresV(audio_path, params=params)])
    X_testH = np.array([extract_featuresH(audio_path, params=params)])
    X_testV = X_testV[:, :, :, np.newaxis]
    X_testH = X_testH[:, :, :, np.newaxis]
    return X_testV, X_testH

def predict(X_testV, X_testH):
    # Load the pre-trained model
    model = load_model('model.h5')
    Y_predictV = model.predict(X_testV)
    Y_predictH = model.predict(X_testH)

    # Get the predicted class label
    prediction = (Y_predictH + Y_predictV)/2
    return prediction

# Define a route to handle file uploads
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        audio_file = request.files['audio_file']

        # file format check
        extension = get_extension(audio_file.filename)
        if extension in ALLOWED_EXTENSIONS:
            # file size check
            if len(audio_file.read()) > MAX_FILESIZE_BYTES:
                audio_file.seek(0)
                return render_template('index.html', error='File size exceeds maximum limit (5 MB).')     
            
            audio_file.seek(0)
            # Save the file to disk
            audio_path = f'static/uploads/{audio_file.filename}'
            audio_file.save(audio_path)
            # Convert the audio file to .aiff format
            convert_to_aiff(audio_path, audio_path)
        else:
            return render_template('index.html', error='Invalid file type. Only WAV and AIFF files are allowed.')
        
        # Preprocess the audio file
        X_testV, X_testH = preprocess(audio_path)
        
        # Make a prediction using the pre-trained model
        prediction = predict(X_testV, X_testH)
        
        # Get the predicted class label
        if prediction > 0.5:
            label = "It's an A-call!! 🎉 ({0:.2f}% possibility)".format(prediction[0][0]*100)
        else:
            label = "Not an A-call ({0:.2f}% possibility)".format((1-prediction[0][0])*100)

        os.remove(audio_path)

        # Render the result
        return render_template('result.html', label=label)
    else:
        return render_template('index.html', error='')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
