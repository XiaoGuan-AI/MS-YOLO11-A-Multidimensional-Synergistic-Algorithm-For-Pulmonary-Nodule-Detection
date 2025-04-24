import torch
import random
import numpy as np
from ultralytics import YOLO

# Enable deterministic settings to ensure that the results of each run are consistent
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(0) # Set random seeds

    # Define data paths and training parameters
    data_yaml_path = 'data/data.yaml'
    model_path = r"D:\StudyRoom\Medical_Imaging\CTB_and_CS\MS-YOLOv11-main\ultralytics\cfg\models\addv11\yolov11n_SMAMCA_Block_Test.yaml"
    epochs = 700
    batch_size = 8
    img_size = 640
    workers = 0  # Win10: workers=0，Win11: workers=4
    device = '0' # Use the first GPU

    # Create a YOLO instance
    model = YOLO(model_path)

    # Start training, and automatically verify after training
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        workers=workers,
        device=device,
        project='runs/detect/train/',
        name='train',
        verbose=True,
        seed=0, # Set random seeds for YOLO training
        determine=True # Enable deterministic calculation
    )
