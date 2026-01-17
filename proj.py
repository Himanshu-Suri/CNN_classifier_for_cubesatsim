import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from scipy import signal
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import seaborn as sns

MODEL_PATH = "signal_classifiermodel.pth"
SAMPLE_RATE = 2.4e6
SPEC_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['aprs', 'fsk', 'noise', 'sstv']
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

print(f"Using device: {DEVICE}\n")

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

def iq_to_spectrogram(iq_samples, nperseg=512, noverlap=256):
    """Convert IQ samples to spectrogram"""
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

def train_model(data_dir, train_ratio=0.8):
  
    print("LOADING AND PREPARING DATA")

    all_files = os.listdir(data_dir)
    npy_files = [f for f in all_files if f.endswith('.npy')]
    
    print(f"Total .npy files found: {len(npy_files)}\n")
    
    if len(npy_files) == 0:
        print("ERROR: No .npy files found in directory!")
        return None
    
    class_files = {'aprs': [], 'fsk': [], 'noise': [], 'sstv': []}
    
    for npy_file in npy_files:
        for class_name in class_files.keys():
            if npy_file.startswith(class_name):
                class_files[class_name].append(npy_file)
                break
    
    print("Files per class:")
    for class_name, files in class_files.items():
        print(f"  {class_name.upper():8s}: {len(files):4d} files")
    print()
    
    train_specs = []
    train_labels = []
    test_specs = []
    test_labels = []
    
    for class_idx, (class_name, files) in enumerate(class_files.items()):
        if len(files) == 0:
            continue
    
        np.random.shuffle(files)
        
        split_point = int(len(files) * train_ratio)
        train_files = files[:split_point]
        test_files = files[split_point:]
        
        print(f"Loading {class_name.upper()}...")
        print(f"  Train: {len(train_files)} files", end="")
        
        for npy_file in train_files:
            filepath = os.path.join(data_dir, npy_file)
            try:
                iq_samples = np.load(filepath)
                spec, _, _ = iq_to_spectrogram(iq_samples)
                train_specs.append(spec)
                train_labels.append(class_idx)
            except Exception as e:
                print(f"\n    ERROR: {npy_file}: {e}")
     
        print(f"  Test:  {len(test_files)} files", end="")
        
        for npy_file in test_files:
            filepath = os.path.join(data_dir, npy_file)
            try:
                iq_samples = np.load(filepath)
                spec, _, _ = iq_to_spectrogram(iq_samples)
                test_specs.append(spec)
                test_labels.append(class_idx)
            except Exception as e:
                print(f"\n    ERROR: {npy_file}: {e}")
     
    
    print(f"\nTotal training samples: {len(train_specs)}")
    print(f"Total test samples: {len(test_specs)}\n")
    
    train_specs = np.array(train_specs)
    train_labels = np.array(train_labels)
    test_specs = np.array(test_specs)
    test_labels = np.array(test_labels)
    
    train_specs_tensor = torch.from_numpy(train_specs).unsqueeze(1).float()
    train_labels_tensor = torch.from_numpy(train_labels).long()
    test_specs_tensor = torch.from_numpy(test_specs).unsqueeze(1).float()
    test_labels_tensor = torch.from_numpy(test_labels).long()
    
    train_dataset = TensorDataset(train_specs_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_specs_tensor, test_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("TRAINING MODEL")

    model = SpectrogramCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_test_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for specs, labels in train_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for specs, labels in test_loader:
                specs, labels = specs.to(DEVICE), labels.to(DEVICE)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Model saved (best test loss: {best_test_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
    
    print("TRAINING COMPLETE - EVALUATING ON TEST SET")

    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for specs, labels in test_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            outputs = model(specs)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predicted_idx.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    predicted_classes = [CLASS_NAMES[i] for i in all_predictions]
    true_classes = [CLASS_NAMES[i] for i in all_true_labels]
    
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    report = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, zero_division=0)
    print("DETAILED CLASSIFICATION REPORT")
    print(report)
 
    
    cm = confusion_matrix(true_classes, predicted_classes, labels=CLASS_NAMES)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Signal Classification (Test Set)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test.png', dpi=100, bbox_inches='tight')
    print(f"Confusion matrix saved to: confusion_matrix_test.png\n")
    plt.show()
    
    return model

def load_model(model_path):
    """Load trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = SpectrogramCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def classify_signal(model, iq_samples):
    """Classify a single signal"""
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
    """Visualize prediction with 4 subplots"""
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
    
    if confidence > 0.8:
        confidence_label = "CONFIDENT"
    elif confidence < 0.6:
        confidence_label = "LOW CONFIDENCE"
    else:
        confidence_label = "MODERATE"
    
    result_text = f"""
    PREDICTED CLASS: {predicted_class.upper()}
    CONFIDENCE: {confidence:.2%}
    
    {confidence_label}
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
    
    print("SIGNAL CLASSIFICATION - TRAIN & PREDICT")

    while True:
        print("\nMAIN MENU:")
        print("1 - Train CNN from scratch (80/20 split)")
        print("2 - Use existing model to classify")
        print("q - Quit")
        
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == "q":
            print("Exiting rn")
            break
        
        elif choice == "1":
            print("TRAINING CNN")
         
            
            data_dir = input("Enter path to directory with .npy files: ").strip()
            
            if not os.path.isdir(data_dir):
                print(f"Error: Directory not found: {data_dir}")
                continue
            
            model = train_model(data_dir, train_ratio=0.8)
            
            if model is not None:
                print("\nTraining complete! Model saved to:", MODEL_PATH)
                print("You can now use option 2 to classify new signals.\n")
        
        elif choice == "2":
            try:
                model = load_model(MODEL_PATH)
            except FileNotFoundError as e:
                print(f"\nError: {e}")
                print("Please train a model first using option 1\n")
                continue
            
            print("SIGNAL CLASSIFICATION")
        
            while True:
                print("\nPrediction Options:")
                print("1 - Classify a single .npy file")
                print("2 - Classify all files in a directory")
                print("b - Back to main menu")
                
                pred_choice = input("\nSelect option: ").strip().lower()
                
                if pred_choice == "b":
                    break
                
                elif pred_choice == "1":
                    filepath = input("Enter path to .npy file: ").strip()
                    
                    if not os.path.exists(filepath):
                        print(f"Error: File not found: {filepath}\n")
                        continue
                    
                    iq_samples = np.load(filepath)
                    predicted_class, confidence, prob_dict, spec = classify_signal(model, iq_samples)
                    
                    print(f"File: {os.path.basename(filepath)}")
                    print(f"Predicted Class: {predicted_class.upper()}")
                    print(f"Confidence: {confidence:.2%}\n")
                    print("All Probabilities:")
                    for cls, prob in prob_dict.items():
                        print(f"  {cls.upper():8s}: {prob:.4f}")
                    
                    fig = visualize_prediction(iq_samples, predicted_class, confidence, prob_dict, spec)
                    plt.savefig(f"prediction_{os.path.basename(filepath).replace('.npy', '.png')}", dpi=100, bbox_inches='tight')
                    print(f"\nVisualization saved to: prediction_{os.path.basename(filepath).replace('.npy', '.png')}")
                    plt.show()
            
                elif pred_choice == "2":
                    directory = input("Enter directory path: ").strip()
                    
                    if not os.path.isdir(directory):
                        print(f"Error: Directory not found: {directory}\n")
                        continue
                    
                    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
                    
                    if not npy_files:
                        print(f"No .npy files found in {directory}\n")
                        continue
                    
                    print(f"\nClassifying {len(npy_files)} files...\n")
                    
                    results = []
                    predictions = []
                    true_labels = []
                    
                    for npy_file in sorted(npy_files):
                        filepath = os.path.join(directory, npy_file)
                        iq_samples = np.load(filepath)
                        predicted_class, confidence, _, _ = classify_signal(model, iq_samples)
                        results.append((npy_file, predicted_class, confidence))
                        predictions.append(predicted_class)
                        
                        for class_name in CLASS_NAMES:
                            if npy_file.startswith(class_name):
                                true_labels.append(class_name)
                                break
                        
                        print(f"  {npy_file}: {predicted_class.upper()} ({confidence:.2%})")
                    
                    print("SUMMARY")
                    for cls in CLASS_NAMES:
                        count = sum(1 for _, pred, _ in results if pred == cls)
                        print(f"{cls.upper():8s}: {count}/{len(results)}")
                    
                    if len(true_labels) == len(predictions):
                        print("CONFUSION MATRIX (Based on filename)")
    
                        accuracy = accuracy_score(true_labels, predictions)
                        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
                        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
                        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
                        
                        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
                        print(f"Precision: {precision:.4f}")
                        print(f"Recall:    {recall:.4f}")
                        print(f"F1 Score:  {f1:.4f}\n")
                        
                        cm = confusion_matrix(true_labels, predictions, labels=CLASS_NAMES)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                                    ax=ax, cbar_kws={'label': 'Count'})
                        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
                        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
                        ax.set_title('Confusion Matrix - Batch Prediction', fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig('confusion_matrix_prediction.png', dpi=100, bbox_inches='tight')
                        print(f"Confusion matrix saved to: confusion_matrix_prediction.png\n")
                        plt.show()
                
                else:
                    print("select 1, 2, or b.\n")
        
        else:
            print("Please select 1, 2, or q.\n")
    
    print("\nDone!")