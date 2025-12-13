from ultralytics import YOLO
import torch

def main():

    # ------------------------------
    # GPU CHECK
    # ------------------------------
    print("\n===== GPU CHECK =====")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("=====================\n")

    # ------------------------------
    # LOAD MODEL (YOLOv12m)
    # ------------------------------
    model = YOLO("yolo12m.pt")   # You can replace with yolo12l.pt for larger model

    # ------------------------------
    # TRAIN CONFIG
    # ------------------------------
    model.train(
        data="data.yaml",        # Path to your dataset YAML
        epochs=120,              # Higher = more accuracy
        imgsz=640,               # Image size
        batch=8,                 # RTX 3050 = best at batch 4–8
        device=0,                # GPU
        optimizer="AdamW",       # Best optimizer for accuracy
        lr0=0.001,               # Learning rate
        amp=True,                # Mixed precision = faster
        augment=True,            # Strong augmentations
        patience=30,             # Early stopping
        pretrained=True          # Improves accuracy
    )

    print("\nTraining Complete!")
    print("Best weights saved at: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
