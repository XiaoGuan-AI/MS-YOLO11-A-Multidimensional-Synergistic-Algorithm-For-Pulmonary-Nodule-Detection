import json
import os

# Read source files
with open(r'E:\yolo\yolov8v10\datasets\data.json', 'r') as file:
    data = json.load(file)

# Process each dictionary
for item in data:
    # If the dictionary contains the 'image_id' key
    if 'image_id' in item:
        # Convert the 'image_id' value to a 5-digit form and fill it with 0 in front
        item['image_id'] = str(item['image_id']).zfill(6)

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# The path to build a new file
new_file_path = os.path.join(current_dir, 'new_data_file.json')

# Write the modified content to a new file
with open(new_file_path, 'w') as file:
    json.dump(data, file, indent=4)