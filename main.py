import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# DATA AUGMENTATION

def augment_audio(audio, sr):
    noise = np.random.randn(len(audio))
    audio_noise = audio + 0.005 * noise

    audio_pitch = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)

    return [audio, audio_noise, audio_pitch]


# FEATURE EXTRACTION (STRONG)

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)
        audio = librosa.util.normalize(audio)

        augmented = augment_audio(audio, sr)

        feature_list = []

        for a in augmented:

            # HIGH RESOLUTION MFCC
            mfcc = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=80)

            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            chroma = librosa.feature.chroma_stft(y=a, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(a)
            spectral = librosa.feature.spectral_contrast(y=a, sr=sr)

            features = np.hstack([
                np.mean(mfcc, axis=1),
                np.mean(delta, axis=1),
                np.mean(delta2, axis=1),
                np.mean(chroma, axis=1),
                np.mean(zcr, axis=1),
                np.mean(spectral, axis=1)
            ])

            feature_list.append(features)

        return feature_list

    except:
        return None


# LOAD DATASET

def load_dataset(data_path="dataset"):
    X, y = [], []
    categories = ["human", "machine"]

    for label, category in enumerate(categories):
        folder = os.path.join(data_path, category)
        files = [f for f in os.listdir(folder) if f.endswith((".wav", ".mp3"))]

        print(f"\nProcessing {category} ({len(files)} files)")

        for file in files:
            path = os.path.join(folder, file)
            features_list = extract_features(path)

            if features_list is not None:
                for f in features_list:
                    X.append(f)
                    y.append(label)

    return np.array(X), np.array(y)


# TRAIN MODEL

def train_model(X, y):
    print("Class distribution:", np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # HANDLE CLASS IMBALANCE
    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, scaler


# MAIN

if __name__ == "__main__":

    print("Loading dataset...")
    X, y = load_dataset()

    print("Dataset shape:", X.shape)

    if len(X) == 0:
        print("No data found!")
        exit()

    print("Training model...")
    model, scaler = train_model(X, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("✅ Model saved!")