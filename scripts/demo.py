import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
import time
sys.path.append('.')

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args

def expand_canvas(img, new_width, new_height):
    """Expand the canvas of a PIL image with white color."""
    # Create a new image with the desired size and white background
    new_image = Image.new('RGB', (new_width, new_height), color='white')
    
    # Calculate the position to paste the original image
    paste_x = (new_width - img.width) // 2
    paste_y = (new_height - img.height) // 2

    # Paste the original image onto the new canvas
    new_image.paste(img, (paste_x, paste_y))

    return new_image

def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)

    inference = Inference(config, args, args.load_path)

    n_imgname = sorted(os.listdir(args.source_dir))
    m_imgname = sorted(os.listdir(args.reference_dir))
    
    img_person = Image.open(os.path.join(args.source_dir, "source_8.jpg")).convert('RGB')
    color_img = Image.open(os.path.join(args.reference_dir, "makeup_1.jpg")).convert('RGB')

    original_width, original_height = img_person.size

    if original_height<original_width:
        new_height = min(800, original_height)
        aspect_ratio = original_height / new_height
        new_width = int(original_width/aspect_ratio)
        img_person.resize((new_width, new_height))
        if new_height<800:
            color_img.resize((new_height, new_height))
        color_img = expand_canvas(color_img, new_width, new_height)
    else:
        new_width = min(800, original_width)
        aspect_ratio = original_width / new_width
        new_height = int(original_height/aspect_ratio)
        img_person.resize((new_width, new_height))
        if new_width<800:
            color_img.resize((new_width, new_width))
        color_img = expand_canvas(color_img, new_width, new_height)

    result = inference.joint_transfer(img_person, color_img, None, None, postprocess=True)
    save_path = os.path.join(args.save_folder, f"result.png")
    result.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model", 
                        default='ckpts/sow_pyramid_a5_e3d2_remapped.pth')

    parser.add_argument("--source-dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--reference-dir", type=str, default="assets/images/makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = "cpu"
    args.device = torch.device("cpu")

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    config = get_config()
    main(config, args)