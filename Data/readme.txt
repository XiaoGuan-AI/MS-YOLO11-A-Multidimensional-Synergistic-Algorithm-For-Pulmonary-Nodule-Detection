This folder contains all the preprocessed data.
The data is authentic and valid.

Use the LUNA16 dataset.
Preprocess images to [−1000, 400] HU window range and convert to RGB .jpg format.
Convert annotations to YOLO format: [class, x_center, y_center, width, height] normalized to [0, 1].
Two dataset versions:
Luna16_Undivided: Original images (recommended)
Luna16_Segmentation: With lung segmentation preprocessing
Split dataset 9:1 into training and validation sets.