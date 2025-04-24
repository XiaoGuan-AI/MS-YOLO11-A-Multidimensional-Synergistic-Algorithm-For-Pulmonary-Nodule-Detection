import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set a unified font size
plt.rcParams.update({
    'font.size': 18, # Global font size
    'axes.titlesize': 18, # Axis title font size
    'axes.labelsize': 18, # Axis label font size
    'xtick.labelsize': 18, # x-axis tick font size
    'ytick.labelsize': 18, # y-axis tick font size
    'legend.fontsize': 18, # Legend font size
})

def smooth_curve(data, factor=0.98):
    """
    Use maximum value retention for curve smoothing
    :param data: raw data
    :param factor: Smoothing degree (this parameter no longer affects calculation)
    :return: smoothed data
    """
    smoothed_data = []
    prev = data[0]
    for point in data:
        smoothed_point = max(prev, point) # Get the maximum value
        smoothed_data.append(smoothed_point)
        prev = smoothed_point
    return np.array(smoothed_data)


if __name__ == '__main__':
    smooth_factor = 0.98
    markevery = 25 # Set symbol interval
    save_dir = "Figure/mAP50"
    os.makedirs(save_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(save_dir) if f.startswith("map50_") and f.endswith(".png")]
    existing_numbers = sorted(
        [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()])
    next_number = (existing_numbers[-1] + 1) if existing_numbers else 1
    save_path = os.path.join(save_dir, f"map50_{next_number}.png").replace("\\", "/")

    result_dict = {
        'YOLO11': r'C:\Users\Administrator\Desktop\LUNA16\910\Undivided\Ablation_Experiment\YOLO11_mAP=79.52%\train6\results.csv',
        'MCA-YOLO11': r'C:\Users\Administrator\Desktop\LUNA16\910\Undivided\Ablation_Experiment\MCA-YOLO11_mAP=80.24%\train25\results.csv',
        'SMAT-YOLO11': r'C:\Users\Administrator\Desktop\LUNA16\910\Undivided\Ablation_Experiment\SMAT-YOLO11_mAP=82.30%\train6\results.csv',
        'MS-YOLO11': r'C:\Users\Administrator\Desktop\LUNA16\910\Undivided\Ablation_Experiment\MS-YOLO11_mAP=83.66%\train8\results.csv',
    }

    markers = ['o', 's', '^', 'd', '*', 'x', 'v', 'p', 'h', '+']
    plt.figure(figsize=(10, 6))

    max_epochs = 320

    for idx, (modelname, res_path) in enumerate(result_dict.items()):
        data = pd.read_csv(res_path, usecols=[6]).values.ravel()
        data = data[:max_epochs]
        smoothed_data = smooth_curve(data, factor=smooth_factor)
        x = np.arange(len(data))

        marker_style = markers[idx % len(markers)]

        x_marked = x[::markevery]
        y_marked = smoothed_data[::markevery]
        plt.plot(x_marked, y_marked, label=f'{modelname}', linewidth=1, color=f'C{idx}',
                 marker=marker_style, markevery=markevery)
        plt.scatter(x_marked, y_marked, marker=marker_style, s=20, color=f'C{idx}')

    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    print(f"mAP50 image saved to: {save_path}")
    plt.show()
