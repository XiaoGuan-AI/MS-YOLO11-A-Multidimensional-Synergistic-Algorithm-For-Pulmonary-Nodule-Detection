import os
import random
from shutil import copy2

def data_set_split(src_images_folder, src_labels_folder, target_data_folder, slice_data=[0.8, 0.1, 0.1]):
    """
    Read the source data folder and generate divided folders, which are divided into three folders: train, valid, and test
    :param src_images_folder: The folder path for image storage
    :param src_labels_folder: The folder path stored by the tag
    :param target_data_folder: The path to the target data folder
    :param slice_data: data division ratio [training set ratio, validation set ratio, test set ratio]
    :return: None
    """
    print("Start dataset division")

    # Get all pictures and tag files and sort them by file name
    image_files = sorted([f for f in os.listdir(src_images_folder) if f.endswith(('.jpg', '.png'))])
    label_files = sorted([f for f in os.listdir(src_labels_folder) if f.endswith('.txt')])

    # Make sure the number of pictures and tags matches
    image_names = set(os.path.splitext(f)[0] for f in image_files)
    label_names = set(os.path.splitext(f)[0] for f in label_files)
    common_names = list(image_names & label_names)

    if len(common_names) == 0:
        print("Error: No matching picture and label was found! Please check whether the file name corresponds to it")
        return

    # Sort by file name and match
    common_names.sort()
    paired_data = [(name + '.jpg', name + '.txt') for name in common_names] # Here assuming the image is .jpg

    # Disrupt data
    random.shuffle(paired_data)

    # Calculate the index
    total = len(paired_data)
    train_stop = int(total * slice_data[0])
    val_stop = int(total * (slice_data[0] + slice_data[1])) if slice_data[1] > 0 else train_stop

    # Divide datasets
    datasets = {
        "train": paired_data[:train_stop],
        "valid": paired_data[train_stop:val_stop] if slice_data[1] > 0 else [],
        "test": paired_data[val_stop:] if slice_data[2] > 0 else []
    }

    # Create a target folder
    for split_name in datasets.keys():
        os.makedirs(os.path.join(target_data_folder, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_data_folder, split_name, "labels"), exist_ok=True)

    # Copy the file
    for split_name, data in datasets.items():
        for img_file, label_file in data:
            copy2(os.path.join(src_images_folder, img_file), os.path.join(target_data_folder, split_name, "images", img_file))
            copy2(os.path.join(src_labels_folder, label_file), os.path.join(target_data_folder, split_name, "labels", label_file))

    # Output division status
    print(f"Data division is completed, total {total} group data")
    print(f"training set: {len(datasets['train'])} group")
    print(f"Verification Set: {len(datasets['valid'])} Group")
    print(f"test set: {len(datasets['test'])} group")


if __name__ == '__main__':
    # Source data path for pictures and tags
    src_images_folder = r"C:\Users\Administrator\Desktop\CS\Undivided\dataset\images"
    src_labels_folder = r"C:\Users\Administrator\Desktop\CS\Undivided\dataset\labels"

    # Target data folder
    target_data_folder = r"C:\Users\Administrator\Desktop\CS\Undivided\split_dataset"

    # Call the function, divide the ratio to 90% training set, 10% verification set, 0% test set
    data_set_split(src_images_folder, src_labels_folder, target_data_folder, slice_data=[0.8, 0.1, 0.1])
