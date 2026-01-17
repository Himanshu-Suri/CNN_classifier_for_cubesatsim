import numpy as np
import os
from datetime import datetime
#this is to generate noise because i was not sure how to capture noise 

OUT_DIR = "../data/real_signals"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_RATE = 2.4e6
DURATION = 2.0          
NOISE_STD = 0.5         

total_samples = int(SAMPLE_RATE * DURATION)

print("Generating synthetic IQ noise...")

for i in range(NUM_SAMPLES):
    noise_i = np.random.normal(0, NOISE_STD, total_samples)
    noise_q = np.random.normal(0, NOISE_STD, total_samples)

    iq_noise = noise_i + 1j * noise_q
    iq_noise = iq_noise.astype(np.complex64)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"noise_{timestamp}_{i:02d}.npy"
    path = os.path.join(OUT_DIR, filename)

    np.save(path, iq_noise)
    print(f"Saved {path}")

print("Noise generation complete.")
