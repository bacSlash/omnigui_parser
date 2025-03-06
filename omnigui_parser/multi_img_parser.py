import os
import csv
from tkinter import filedialog, Tk
import argparse
from PIL import Image
import torch
from .utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
from pathlib import Path

# Set device for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to search for the model in parent directories
def find_model_path(model_filename='best.pt', search_folder='weights/icon_detect'):
    current_dir = Path(__file__).resolve.parent # Start from the script directory
    
    while current_dir !=current_dir.root: # Traverse upwards till root
        model_path = current_dir / search_folder / model_filename
        if model_path.exists():
            return str(model_path) # Return the absolute path if found
        current_dir = current_dir.parent # Move one level up

    raise FileNotFoundError(f"Model file '{model_filename}' not found in any parent directory.")

# Initialize models
ICON_DETECT_MODEL_PATH = find_model_path()
ICON_CAPTION_MODEL_NAME = 'florence2'
ICON_CAPTION_MODEL_PATH = 'microsoft/Florence-2-base'

# Load YOLO model for icon detection
yolo_model = get_yolo_model(ICON_DETECT_MODEL_PATH)
caption_model_processor = get_caption_model_processor(ICON_CAPTION_MODEL_NAME, ICON_CAPTION_MODEL_PATH)

def process_image(image_path):
    image = Image.open(image_path)
    box_threshold = 0.05
    iou_threshold = 0.1
    use_paddleocr = False
    imgsz = 1920
    icon_process_batch_size = 64
    
    ocr_bbox_rslt, _ = check_ocr_box(
        image_path,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold': 0.9},
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt
    
    _, _, parsed_content_list = get_som_labeled_img(
        image_path,  
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config={},
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        batch_size=icon_process_batch_size
    )
    
    return parsed_content_list

def select_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder with images")
    return folder_path

def process_folder(folder_path, output_csv_path):
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Parser Content'])
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                try:
                    parsed_content = process_image(image_path)
                    writer.writerow([filename, str(parsed_content)])
                    print(f'Processed {filename}')
                except Exception as e:
                    print(f'Failed to process {filename}: {str(e)}')
                    
def process_folder_cli():
    parser = argparse.ArgumentParser(description='Process images from folder.')
    parser.add_argument('folder_path', type=str, help='Path to folder with images')
    parser.add_argument('output_csv_path', type=str, help='Path to save output CSV file')
    args = parser.parse_args()
    
    process_folder(args.folder_path, args.output_csv_path)
                    
def main():
    folder_path = select_folder()
    if folder_path:
        output_csv_path = os.path.join(folder_path, 'parsed_output.csv')
        process_folder(folder_path, output_csv_path)
        print('All images processed. Results saved to', output_csv_path)
    else:
        print('No folder selected. Exiting.')

if __name__ == "__main__":
    main()
