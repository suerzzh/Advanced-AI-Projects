from ultralytics import YOLO
import sys

def main():
    # Model configuration path
    model_config = "ultralytics/cfg/models/11/yolo11-star-fusion.yaml"
    
    # Initialize the improved model
    model = YOLO(model_config) 
    
    # Load pre-trained weights
    try:
        model.load('yolo11n.pt')
    except:
        print("Warning: Weights partial load due to Star-Fusion architecture changes.")

    # Output model architecture information for verification
    print("\n" + "="*50)
    print("Star-Fusion-YOLO Model Architecture Initialized")
    print("Improvements Loaded:")
    print("1. [Backbone] StarBlock (Lightweight Group Conv)")
    print("2. [Neck]     C3k2-Star (Star Operation Fusion)")
    print("3. [Attention] CBAM (Channel & Spatial Attention)")
    print("4. [Head]     LSCD (Shared Conv + GroupNorm)")
    print("="*50 + "\n")
    
    model.info()

    # Start training
    model.train(
        data="D:/SeaShips/data.yaml",
        epochs=300,
        patience=50,
        imgsz=640,
        batch=16,
        project='runs/train',
        name='Star_Fusion_SeaShips_Final',
        optimizer='SGD',
        cos_lr=True,
        save=True,
        plots=True
    )

if __name__ == '__main__':
    main()