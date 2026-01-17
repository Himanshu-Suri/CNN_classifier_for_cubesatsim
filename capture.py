from rtlsdr import RtlSdr
import numpy as np
import os
import time
import glob
from datetime import datetime

SAVE_DIR = "../data/real_signals"
os.makedirs(SAVE_DIR, exist_ok=True)

CENTER_FREQ = 434.9e6
SAMPLE_RATE = 2.4e6
CHUNK_SIZE = 256_000

DURATION_MAP = {
    "aprs": 2.0,
    "fsk":  2.0,
    "sstv": 15.0,
}

TARGET_SAMPLES = {
    "aprs": 40,
    "fsk":  40,
    "sstv": 15,
}

POST_CAPTURE_SLEEP = {
    "aprs": 30.0,   
    "fsk":  0.5,
    "sstv": 1.0,
}

SIGNAL_TYPES = list(TARGET_SAMPLES.keys())

def count_existing(signal):
    return len(glob.glob(f"{SAVE_DIR}/{signal}_*.npy"))

def read_iq_samples(sdr, total_samples):
    samples = []
    remaining = total_samples

    while remaining > 0:
        n = min(CHUNK_SIZE, remaining)
        samples.append(sdr.read_samples(n))
        remaining -= n

    return np.concatenate(samples)

sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = 30   

while True:
    print("\nCurrent dataset status:")
    for s in SIGNAL_TYPES:
        print(f"  {s.upper():6s}: {count_existing(s)}/{TARGET_SAMPLES[s]}")

    print("\nChoose mode:")
    for i, s in enumerate(SIGNAL_TYPES, start=1):
        print(f"{i} → {s.upper()}")
    print("quit")

    choice = input("Select: ").strip().lower()

    if choice == "q":
        break

    if not choice.isdigit() or int(choice) not in range(1, len(SIGNAL_TYPES) + 1):
        print("Invalid choice.")
        continue

    signal_name = SIGNAL_TYPES[int(choice) - 1]

    if count_existing(signal_name) >= TARGET_SAMPLES[signal_name]:
        print(f"{signal_name.upper()} already has enough samples.")
        continue

    duration = DURATION_MAP[signal_name]
    total_samples = int(SAMPLE_RATE * duration)
    sleep_time = POST_CAPTURE_SLEEP[signal_name]

    print(f"\nCUBESATSIM MODE: {signal_name.upper()}")
    input("Press ENTER when ready to capture...")
    print("Capturing datat right now\n")

    while count_existing(signal_name) < TARGET_SAMPLES[signal_name]:
        try:
            
            sdr.read_samples(1024)

            iq_samples = read_iq_samples(sdr, total_samples)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{SAVE_DIR}/{signal_name}_{timestamp}.npy"
            np.save(filename, iq_samples)

            print(f"Saved → {filename}")
            print(f"Waiting {sleep_time:.1f}s for next transmission\n")

            time.sleep(sleep_time)

        except Exception as e:
            print(f"Error: {e}")
            break
    print(f"\n{signal_name.upper()} collection complete.")

sdr.close()
print("\nended.")
