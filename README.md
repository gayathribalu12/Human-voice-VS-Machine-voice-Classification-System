# Human-voice-VS-Machine-voice-Classification-System Abstract


This project aims to develop a machine learning-based system that classifies the input audio signal as a human voice or a machine voice.

Audio signals are pre-processed and converted into **Mel Frequency Cepstral Coefficient (MFCC) and Mel Spectrogram features**, which include the perceptual characteristics of the speech signal.

The system uses **XGBoost (Extreme Gradient Boosting)**, a fast and efficient learning technique that uses the gradient boosting framework, to classify the features of the audio signal. XGBoost is highly efficient in learning non-linear patterns in the audio signal and classifies the signal with high accuracy.

The system has **achieved a high accuracy of 92.33%**, which shows that the system has the ability to differentiate between synthetic voices and natural human voices.

Furthermore, the system has a **decision layer that uses confidence levels** to increase the accuracy of the predictions:
-   **High Confidence (Above 0.85)**
-   **Needs Review (Between 0.65 and 0.85)**
-   **Uncertain (Below 0.65)**


##  Dataset

Due to large file size, the dataset is hosted externally.

 Download here:  
 https://drive.google.com/https://drive.google.com/drive/folders/15CJwccNoxUGNMc_4-g9yfJpbGSsU0DD8?usp=drive_link

##  Dataset Setup

1. Download dataset from the link above  
2. Extract the files  
3. Place them in:
dataset/
4. Run the project

### Dataset Details:
- Format: `.mp3`
- Sampling Rate: 16 kHz
- Channels: Mono
- Classes:
  - Human Speech
  - Machine-Generated Speech
- Preprocessing:
  - Noise reduction
  - Silence trimming
  - Standardization of duration


##  Features & Methodology

###  Audio Preprocessing
- Resampling to 16 kHz
- Conversion to mono-channel WAV
- Noise reduction & silence removal

###  Feature Extraction
- MFCC (Mel-Frequency Cepstral Coefficients)
- Mel Spectrogram

###  Model Used
- **XGBoost Classifier**
  - Handles non-linear relationships
  - High efficiency and scalability
  - Built-in regularization to prevent overfitting



##  Model Performance   |

 Accuracy --->**92.33%**
 Model --> XGBoost    



##  Decision System (Intelligent Layer)

 Confidence Score   Decision 

 ≥ 0.85            High Confidence 
 0.65 – 0.85       Needs Review 
 < 0.65            Uncertain 
