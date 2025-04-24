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

def smooth_curve(data, factor=0.9):
    """
    Curve smoothing using exponentially weighted moving average (EWMA)
    :param data: raw data
    :param factor: Smoothing degree, range [0, 1], the larger the value, the smoother it is
    :return: smoothed data
    """
    smoothed_data = []
    prev = data[0]
    for point in data:
        smoothed_point = factor * prev + (1 - factor) * point
        smoothed_data.append(smoothed_point)
        prev = smoothed_point
    return np.array(smoothed_data)

if __name__ == '__main__':
    smooth_factor = 0.8
    markevery = 25
    max_epochs = 300
    save_dir = "Figure/Loss"
    os.makedirs(save_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(save_dir) if f.startswith("loss_") and f.endswith(".png")]
    existing_numbers = sorted(
        [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()])
    next_number = (existing_numbers[-1] + 1) if existing_numbers else 1
    save_path = os.path.join(save_dir, f"loss_{next_number}.png").replace("\\", "/")

    result_dict = {
        'YOLOv10': r'C:\Users\Administrator\Desktop\LUNA16\910\Undivided\Comparative_Experiment\YOLOv10_mAP=75.44%\train10\results.csv',
        'YOLO11': r'C:\Users\Administrator\Desktop\LUNA16\910\Undivided\Comparative_Experiment\YOLO11_mAP=79.52%\train6\results.csv',
        'YOLO12': r'C:\Users\Administrator\Desktop\LUNA16\910\Undivided\Comparative_Experiment\YOLO12_mAP=77.76%\train\results.csv',
        'MS-YOLO11': r'C:\Users\Administrator\Desktop\LUNA16\910\Undivided\Comparative_Experiment\MS-YOLO11_mAP=83.66%\train8\results.csv',
        'MSDet': r'C:\Users\Administrator\Desktop\LUNA16\910\Undivided\Comparative_Experiment\MSDet_mAP=78.17%\train\results.csv',
    }

    markers = ['o', 's', '^', 'd', 'v', 'x', '*', 'p', 'h', '+']
    plt.figure(figsize=(10, 6))

    for idx, (modelname, res_path) in enumerate(result_dict.items()):
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            box_loss = pd.read_csv(res_path, usecols=[1]).values.ravel()
            obj_loss = pd.read_csv(res_path, usecols=[2]).values.ravel()
            cls_loss = pd.read_csv(res_path, usecols=[3]).values.ravel()
            data = np.round(box_loss + obj_loss + cls_loss, 5)
        else:
            with open(res_path, 'r') as f:
                data = np.array([float(d.strip().split()[5]) for d in f.readlines()])

        data = data[:max_epochs]
        smoothed_data = smooth_curve(data, factor=smooth_factor)
        x = np.arange(len(data))

        marker_style = markers[idx % len(markers)]
        plt.plot(x, smoothed_data, label=f'{modelname}', linewidth=1,
                 marker=marker_style, markersize=4, markevery=markevery)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    print(f"Loss image saved to: {save_path}")
    plt.show()
