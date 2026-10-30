from ultralytics import YOLO
import torch

def main():
    # --- Configuration ---
    # Path to your data.yaml file which defines your dataset splits and classes
    data_config_path = 'data.yaml'

    # Number of epochs to train for. Start with a smaller number on a CPU.
    # Training will be slow without a GPU. 25 epochs is a good baseline test.
    num_epochs = 25

    # Choose a model to start from. yolov12n.pt (Nano) is the smallest and fastest YOLOv12 model.
    # It is a highly practical choice for edge deployment and rapid training.
    model_name = 'yolov12n.pt'
    
    # ---------------------

    print("--- Starting Advanced Pothole Detection Model Training ---")
    print(f"Using compute device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Training for {num_epochs} epochs with the next-generation {model_name} architecture.")
    print(f"Data configuration mapped to: {data_config_path}")
    print("----------------------------------------------------------")

    # Load the YOLOv12 model
    # The ultralytics library will automatically pull the yolov12n.pt weights if not cached locally.
    model = YOLO(model_name)

    # Train the model
    # The 'data' argument points to your .yaml file.
    # The 'epochs' argument specifies how many times to iterate through the dataset.
    results = model.train(
        data=data_config_path,
        epochs=num_epochs,
        imgsz=640,  # 640x640 is the optimal standardized input resolution for this architecture
        project='runs/train', # Root directory to save training runs
        name='pothole_detection_yolov12_run1' # Updated folder name to reflect the v12 upgrade
    )

    print("--- Training Sequence Complete ---")
    print("Model and analytic results successfully saved in the 'runs/train/pothole_detection_yolov12_run1' directory.")
    print("Your optimal deployment model is saved as 'best.pt' inside the weights subfolder.")

if __name__ == '__main__':
    main()