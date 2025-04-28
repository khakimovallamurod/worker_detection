import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV faylni o‘qish
csv_path = "/home/xakimov-allamurod/Documents/computer-vision/worker_detection/train10/results.csv"
df = pd.read_csv(csv_path)

# 2. Train va Val Loss (umumiy) grafigi
df['train/loss'] = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
df['val/loss'] = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train/loss'], label='Train Loss', linewidth=3)
plt.plot(df['epoch'], df['val/loss'], label='Validation Loss', linewidth=3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train va Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('train_val_loss.png')
plt.show()

# 3. mAP50 va Precision grafigi (agar Top-1 accuracy bo‘lmasa)
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=3)
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=3)
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('mAP50 va Precision grafigi')
plt.legend()
plt.grid(True)
plt.savefig('mAP50_precision.png')
plt.show()
