# Whale Call Recognition Flask App

This project is an extension to the whale-call-classification project which allows to recognize Blue whale A-calls by simply uploading a test audio file on the web app. A-Calls are the most common vocalization of Blue whales and play an important role in their communication and behavior. The model was trained on the dataset of 26,000 whale call audio files and tested on a thousand audio files. The model was able to achieve an accuracy of ~96% on the test dataset.

## Installation

Clone the repo and cd into it. Make a virtual environment if you prefer and install the required packages by running:
```
pip install -r requirements.txt
```

## Usage

Start the server by running:
```
python app.py
```

The web app will be available at ```http://localhost:8080/``` where you can upload an audio file to test the model.
