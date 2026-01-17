# RF Signal Classification using CNN

A complete deep learning pipeline for classifying radio frequency (RF) signals into 4 categories: **APRS**, **FSK**, **Noise**, and **SSTV** using a Convolutional Neural Network (CNN).


## Requirements

```bash
pip install numpy torch scipy scikit-learn matplotlib seaborn
```

## Data Format

Your `.npy` files should be named with the class prefix:

```
data_directory/
├── aprs_2026-01-06_16-46-43.npy
├── aprs_2026-01-07_10-23-15.npy
├── fsk_2026-01-06_09-12-34.npy
├── fsk_2026-01-07_11-45-22.npy
├── noise_2026-01-06_12-30-45.npy
├── noise_2026-01-07_08-15-30.npy
├── sstv_2026-01-06_15-20-10.npy
└── sstv_2026-01-08_09-45-55.npy
```

**Important**: Each file must start with the class name (`aprs`, `fsk`, `noise`, or `sstv`)

Each `.npy` file should contain **complex IQ samples** (1D array of complex numbers):
```python
iq_samples = np.array([0.5+0.3j, -0.2+0.8j, ...]) 
```

## Quick Start

### 1. Train a Model

```bash
python signal_classifier.py
```

```
MAIN MENU:
1 - Train CNN from scratch (80/20 split)
2 - Use existing model to classify
q - Quit

Select option: 1

Enter path to directory with .npy files:
/path/to/your/data

# Training starts automatically...
```

**Output:**
- Trained model saved to `signal_classifier_model.pth`
- Test metrics displayed (Accuracy, Precision, Recall, F1)
- Confusion matrix saved as `confusion_matrix_test.png`

### 2. Make Predictions

```
Select option: 2

Prediction Options:
1 - Classify a single .npy file
2 - Classify all files in a directory
b - Back to main menu

Select option: 2

Enter directory path:
/path/to/your/data
```

**Output:**
- Predictions for all files
- Accuracy, Precision, Recall, F1 Score
- Confusion matrix saved as `confusion_matrix_prediction.png`
- ndividual prediction visualizations (PNG files)

## Model Architecture

```
SpectrogramCNN
├── Feature Extraction (Conv2D blocks)
│   ├── Conv2D(1 → 32) → ReLU → MaxPool
│   ├── Conv2D(32 → 64) → ReLU → MaxPool → Dropout
│   └── Conv2D(64 → 128) → ReLU → MaxPool
├── Global Average Pooling (16×16 → 4×4)
└── Classification (Dense layers)
    ├── Linear(2048 → 256) → ReLU → Dropout
    └── Linear(256 → 4) → Output logits
```

## Configuration

Edit these parameters in the script:

```python
BATCH_SIZE = 16              
LEARNING_RATE = 0.001       
NUM_EPOCHS = 50             
SAMPLE_RATE = 2.4e6         
SPEC_SIZE = 128              
```

## Signal Processing Pipeline

1. **Load IQ Samples**: Complex-valued RF signal (1D array)
2. **STFT**: Convert to time-frequency domain (Spectrogram)
3. **Normalization**: Frequency filtering & power normalization
4. **Resizing**: Scale to 128×128 fixed size
5. **Classification**: Pass through CNN → Get class probabilities

## Training Details

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam (adaptive learning rate)
- **Early Stopping**: Stops if validation loss doesn't improve for 10 epochs
- **Data Split**: 80% training, 20% testing (per class)
- **Regularization**: Dropout layers prevent overfitting

## Output Files

| File | Description |
|------|-------------|
| `signal_classifier_model.pth` | Trained model weights |
| `confusion_matrix_test.png` | Confusion matrix on test set |
| `confusion_matrix_prediction.png` | Confusion matrix on predictions |
| `prediction_*.png` | 4-panel visualization per file |

## Performance Metrics

After training and prediction, you'll see:

- **Accuracy**: Overall correctness percentage
- **Precision**: When we predict class X, how often is it correct?
- **Recall**: When class X occurs, how often do we find it?
- **F1 Score**: Harmonic mean of Precision and Recall and also a Confusion Matrix

- Check file location and naming convention
- Files must start with class name: `aprs_`, `fsk_`, `noise_`, `sstv_`

# CNN_classifier_for_cubesatsim
