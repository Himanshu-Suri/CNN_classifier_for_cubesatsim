#this is to check teh number of samples i capttured earlier using sdr
import os

source_dir = "/home/himanshu/Desktop/rf-signal-classification/data/real_signals"

all_files = os.listdir(source_dir)
npy_files = [f for f in all_files if f.endswith('.npy')]

class_counts = {
    'aprs': 0,
    'fsk': 0,
    'noise': 0,
    'sstv': 0
}

for f in npy_files:
    for class_name in class_counts.keys():
        if f.startswith(class_name):
            class_counts[class_name] += 1
            break

print(f"\nTotal .npy files: {len(npy_files)}\n")
for cls, count in class_counts.items():
    print(f"{cls.upper():8s}: {count:3d} files")
print()