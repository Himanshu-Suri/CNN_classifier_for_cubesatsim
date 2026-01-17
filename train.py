import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os

MODEL_PATH = "signal_classifier_model.pth"
SAMPLE_RATE = 2.4e6
SPEC_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['aprs', 'fsk', 'noise', 'sstv']

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SpectrogramCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = SpectrogramCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def iq_to_spectrogram(iq_samples, nperseg=512, noverlap=256):
    f, t, Sxx = signal.spectrogram(
        iq_samples,
        fs=SAMPLE_RATE,
        nperseg=nperseg,
        noverlap=noverlap,
        window='hann'
    )
    
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    freq_mask = np.abs(f) < 500e3
    Sxx_plot = Sxx_db[freq_mask, :]
    Sxx_plot = (Sxx_plot - np.min(Sxx_plot)) / (np.max(Sxx_plot) - np.min(Sxx_plot) + 1e-10)
    zoom_factors = (SPEC_SIZE / Sxx_plot.shape[0], SPEC_SIZE / Sxx_plot.shape[1])
    spec_resized = zoom(Sxx_plot, zoom_factors, order=1)
    
    return spec_resized.astype(np.float32), f, t

def classify_signal(model, iq_samples, confidence_threshold=0.5):
    spec, f, t = iq_to_spectrogram(iq_samples)
    spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(spec_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence = confidence.item()
    prob_dict = {CLASS_NAMES[i]: probabilities[i].item() for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence, prob_dict, spec

def visualize_prediction(iq_samples, predicted_class, confidence, prob_dict, spec):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    i_part = np.real(iq_samples[:2000])
    q_part = np.imag(iq_samples[:2000])
    ax1.plot(i_part, label='I (Real)', alpha=0.7, linewidth=1)
    ax1.plot(q_part, label='Q (Imag)', alpha=0.7, linewidth=1)
    ax1.set_title('IQ Time Series (first 2000 samples)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, :])
    im = ax2.imshow(spec, aspect='auto', cmap='viridis', origin='lower')
    ax2.set_title('Spectrogram', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency')
    plt.colorbar(im, ax=ax2, label='Power (dB)')
    
    ax3 = fig.add_subplot(gs[2, 0])
    classes = list(prob_dict.keys())
    probs = list(prob_dict.values())
    colors = ['green' if c == predicted_class else 'lightgray' for c in classes]
    ax3.barh(classes, probs, color=colors)
    ax3.set_xlabel('Probability')
    ax3.set_title('Classification Probabilities', fontsize=12, fontweight='bold')
    ax3.set_xlim([0, 1])
    for i, (cls, prob) in enumerate(zip(classes, probs)):
        ax3.text(prob + 0.02, i, f'{prob:.3f}', va='center', fontweight='bold')
    
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    result_text = f"""
    PREDICTED CLASS: {predicted_class.upper()}
    CONFIDENCE: {confidence:.2%}
    
    {"✓ CONFIDENT" if confidence > 0.8 else "⚠ LOW CONFIDENCE" if confidence < 0.6 else "◐ MODERATE"}
    """
    ax4.text(
        0.5, 0.5, result_text,
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace'
    )
    
    plt.suptitle('Signal Classification Result', fontsize=16, fontweight='bold', y=0.995)
    return fig

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    
    print("\n" + "="*70)
    print("SIGNAL CLASSIFICATION INFERENCE")
    print("="*70 + "\n")
    
    while True:
        print("\nOptions:")
        print("1 - Classify a single .npy file")
        print("2 - Classify all files in a directory")
        print("q - Quit")
        
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == "q":
            print("Exiting...")
            break
        
        elif choice == "1":
            filepath = input("Enter path to .npy file: ").strip()
            
            if not os.path.exists(filepath):
                print(f"file not found: {filepath}")
                continue
            
            iq_samples = np.load(filepath)
            predicted_class, confidence, prob_dict, spec = classify_signal(model, iq_samples)
            
            print(f"\n{'='*70}")
            print(f"File: {os.path.basename(filepath)}")
            print(f"{'='*70}")
            print(f"Predicted Class: {predicted_class.upper()}")
            print(f"Confidence: {confidence:.2%}\n")
            print("All Probabilities:")
            for cls, prob in prob_dict.items():
                print(f"  {cls.upper():8s}: {prob:.4f}")
            
            fig = visualize_prediction(iq_samples, predicted_class, confidence, prob_dict, spec)
            plt.savefig(f"prediction_{os.path.basename(filepath).replace('.npy', '.png')}", dpi=100)
            plt.show()
        
        elif choice == "2":
            directory = input("Enter directory path: ").strip()
            
            if not os.path.isdir(directory):
                print(f" Directory not found: {directory}")
                continue
            
            npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
            
            if not npy_files:
                print(f"NoA .npy files found in {directory}")
                continue
            
            print(f"\nClassifying {len(npy_files)} files...\n")
            
            results = []
            for npy_file in sorted(npy_files):
                filepath = os.path.join(directory, npy_file)
                iq_samples = np.load(filepath)
                predicted_class, confidence, _, _ = classify_signal(model, iq_samples)
                results.append((npy_file, predicted_class, confidence))
                print(f"  {npy_file}: {predicted_class.upper()} ({confidence:.2%})")
            
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            for cls in CLASS_NAMES:
                count = sum(1 for _, pred, _ in results if pred == cls)
                print(f"{cls.upper():8s}: {count}/{len(results)}")
        
        else:
            print("Invalid choice.")
    
    print("\nDone!")
