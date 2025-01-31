import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from librosa.feature import mfcc
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox

model = None
result_label = None
emotion_labels = ['angry', 'happy', 'neutral', 'sad']  

def record_audio(filename="output.wav", duration=5, fs=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, audio_data)
    print(f"Recording saved to {filename}")
    return filename

def load_audio(filepath, sr=22050):
    audio, sample_rate = librosa.load(filepath, sr=sr)
    return audio, sample_rate

def extract_features(audio, sr):
    mfccs = mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

try:
    model = load_model("age_emotion_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_age_emotion(features):
    if model is None:
        messagebox.showerror("Model Error", "Failed to load the model.")
        return None
    features = np.expand_dims(features, axis=0)  
    prediction = model.predict(features)
    return prediction

def on_predict():
    filename = filedialog.askopenfilename(title="Select an audio file", filetypes=[("Audio Files", "*.wav *.mp3")])
    if not filename:
        messagebox.showwarning("No file selected", "Please select a valid audio file.")
        return
    try:
        audio, sr = load_audio(filename)
        features = extract_features(audio, sr)
        predictions = predict_age_emotion(features)

        if predictions is None:
            return

        age = int(predictions[0][0])  
        emotion_idx = np.argmax(predictions[0][1:])
        emotion = emotion_labels[emotion_idx]

        result_label.config(text=f"Detected Age: {age}, Emotion: {emotion}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def setup_gui():
    root = tk.Tk()
    root.title("Age and Emotion Detection")
    btn_predict = tk.Button(root, text="Select Audio File", command=on_predict)
    btn_predict.pack(pady=10)
    global result_label 
    result_label = tk.Label(root, text="Results will appear here.", font=("Arial", 14))
    result_label.pack(pady=20)
    root.mainloop()
setup_gui()