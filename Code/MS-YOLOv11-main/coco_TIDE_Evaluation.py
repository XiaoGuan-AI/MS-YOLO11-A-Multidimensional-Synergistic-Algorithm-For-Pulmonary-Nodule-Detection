import warnings
import os
import argparse
import sys
import io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str,
                        default=r'D:\StudyRoom\Medical_Imaging\CTB_and_CS\YOLOv11-main\COCO\data.json',
                        help='label coco json path')
    parser.add_argument('--pred_json', type=str,
                        default=r'D:\StudyRoom\Medical_Imaging\CTB_and_CS\YOLOv11-main\COCO\new_pre_file.json',
                        help='pred coco json path')
    return parser.parse_known_args()[0]


def get_next_val_directory(base_path):
    """Find the next available directory (val, val2, val3, ...)."""
    i = 1
    while True:
        val_dir = os.path.join(base_path, f'val{str(i) if i > 1 else ""}')
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
            return val_dir
        i += 1


def print_coco_metrics(eval, file):
    result_str = ""
    metric_labels = [
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
    ]

    for i, label in enumerate(metric_labels):
        result_str += f"{label} = {eval.stats[i]:.4f}\n"


    print(result_str)
    file.write(result_str + "\n")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json

    # Get the next val directory
    base_val_path = r'D:\StudyRoom\Medical_Imaging\CTB_and_CS\YOLOv11-main\COCO\val'
    val_dir = get_next_val_directory(base_val_path)
    save_path = os.path.join(val_dir, "COCO_TIDE.txt")

    with open(save_path, "w") as file:
        file.write("==================== COCO Evaluation Metrics ====================\n")

        # Initialize the COCO API
        anno = COCO(anno_json)
        pred = anno.loadRes(pred_json)

        # Run COCO evaluation
        eval = COCOeval(anno, pred, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()

        # Write detailed COCO evaluation results
        print_coco_metrics(eval, file)

        file.write("\n==================== TIDE Evaluation Metrics ====================\n")

        # Run TIDE evaluation
        tide = TIDE()
        tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)

        # **Fix key point: Use `sys.stdout` to capture TIDE output**
        tide_output = io.StringIO()
        sys.stdout = tide_output # Redirect standard output
        tide.summarize() # Perform evaluation
        sys.stdout = sys.__stdout__ # Restore standard output
        tide_result = tide_output.getvalue()

        print(tide_result) # Let the result be displayed in the terminal
        file.write(tide_result + "\n") # Write to the file

    print(f"Evaluation results saved to: {save_path}")
