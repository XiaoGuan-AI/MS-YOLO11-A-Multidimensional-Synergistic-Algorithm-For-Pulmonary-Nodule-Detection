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
    # Define the model path and the source data path
    model_path = r'D:\StudyRoom\Medical_Imaging\CTB_and_CS\MS-YOLOv11-main\runs\detect\train\train35\weights\best.pt'
    source_path = 'data/valid/images'

    # Create a YOLO instance
    model = YOLO(model_path)

    # Start prediction
    model.predict(source=source_path,
                  project='runs/detect/predict/',
                  name='predict',
                  save=True,
                  save_txt=True
                  )


