import json
import os

# Read source files
input_file_path = r'C:\Users\Administrator\Desktop\LUNA16\910\Segmentation\Comparative_Experiment\MSDet\val2\predictions.json'
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Process each dictionary
for item in data:
    if 'image_id' in item:
        item['image_id'] = str(item['image_id']).zfill(4)

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# The path to build a new file
new_file_path = os.path.join(current_dir, 'new_pre_file.json')

# Write the modified content to a new file
with open(new_file_path, 'w') as file:
    json.dump(data, file, indent=4)

# Print success message and file path
print("Conversion successful!")
print(f"New file saved at: {new_file_path}")
