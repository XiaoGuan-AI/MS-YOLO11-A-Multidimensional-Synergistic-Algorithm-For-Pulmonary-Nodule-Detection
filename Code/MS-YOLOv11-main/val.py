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
    torch.cuda.manual_seed_all(seed)  # if using多 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(0) # Set random seeds

    # Dataset configuration file
    data_yaml_path = 'data/data.yaml'
    # Model configuration file
    model_path = r'D:\StudyRoom\Medical_Imaging\CTB_and_CS\MS-YOLOv11-main\runs\detect\train\train35\weights\best.pt'

    # Create a YOLO instance
    model = YOLO(model_path)

    # Verify
    model.val(
        data=data_yaml_path,
        split='val',
        imgsz=640,
        batch=8,
        project='runs/detect/val/',
        name='val',
        save_json=True,
        seed=0, # Set a random seed for YOLO verification
        determine=True # Enable deterministic calculation
    )
