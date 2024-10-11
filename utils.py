import csv
import json
from time import time
import concurrent.futures
from tqdm import tqdm
import argparse
import shutil
import numpy as np
import pynvml
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import get_backend
import getpass
import requests
import pickle
import gdown
import pickle
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects


def n_trainable_parameters(model):
    return sum([p.nelement() for p in model.parameters() if p.requires_grad])

def pickle_save(obj, fp):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

def download_gdrive(id, target_path):
    url = f'https://drive.google.com/uc?id={id}&export=download'
    gdown.download(url, str(target_path), quiet=False)

def download(url, target_path, is_zenodo=False):
    target_path = str(target_path)
    if is_zenodo:
        headers = {
            "Host": "zenodo.org",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9,tr;q=0.8,pt;q=0.7",
            "Cookie": "session=3fcf3072c7b1e74d_6703d669.jKaCFq1Oh5Ri7Ht2oNZCMmiyTuo; 5569e5a730cade8ff2b54f1e815f3670=05c63ecff7930aed1994a002942c3fd1; csrftoken=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcyODQ4OTc5NiwiZXhwIjoxNzI4NTc2MTk2fQ.IlRmSVFMZHFJelA0eXhpZlk3cjRCcFZqV0VPTEZKS0M5Ig.RxXe74CvNWqUBNKS6E9PLBKxAkdMxqxlAtDZcibX_jFTWq-77iYM6nNyGxAFV-yX98pvO9vfki0-95ScSMp8yA",
            "Connection": "keep-alive"
        }
    else:
        headers = None
    
    response = requests.get(url, headers=headers, stream=True)

    total_size = int(response.headers.get('content-length', 0))  # Get total file size

    if response.status_code == 200:
        with open(target_path, 'wb') as file, tqdm(
            desc="Downloading", total=total_size, unit='B', unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))
        print(f"File saved to {target_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def is_empty(x):
    if isinstance(x, np.ndarray):
        return x.size == 0
    elif isinstance(x, list):
        return len(x) == 0
    return False

def equidistant_indices(source_length, target_length):
    assert source_length > target_length, 'Source length is less than target length.'
    inds = np.round(np.linspace(0, source_length - 1, num=target_length)).astype(np.int32).tolist()
    assert np.array_equal(inds, np.unique(inds))
    return inds

def normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)

def standardize(tensor, mean_val, std_val):
    return (tensor - mean_val) / std_val

def detach_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        x.requires_grad = False
        x = x.cpu()
        x = x.numpy()
    return x

class Logger:
    def __init__(self, log_path, print_=True, log_=True):
        self.log_path = log_path
        self.print = print_
        self.log = log_

    def __call__(self, s, newline=True):
        # Prints log
        sep = "\n" if newline else ","
        if self.print:
            print(str(s), end=sep, flush=True)
        if self.log:
            with open(self.log_path, 'a+') as f_log:
                f_log.write(str(s) + sep)

class CsvWriter:
    # Save performance as a csv file
    def __init__(self, out_path, fieldnames, input_path=None, debug=False):
        self.output_path = out_path
        self.fieldnames = fieldnames
        self.debug = debug
        if not debug:
            if input_path is None:
                with open(out_path, "w") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
            else:
                try:
                    shutil.copy(input_path, out_path)
                except:
                    with open(out_path, "w") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
    def update(self, performance_dict):
        # add missing values as NaN
        if not self.debug:
            for key in self.fieldnames:
                if key not in performance_dict.keys():
                    performance_dict[key] = np.nan
            with open(self.output_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(performance_dict)

def memory():
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        ret = 'Total: {:.2f} | Free: {:.2f} | Used: {:.2f}'.format(
            info.total / 1000000000,
            info.free / 1000000000,
            info.used / 1000000000,)
    else:
        ret = "CUDA not available"
    return ret
        

def run_parallel(func, my_iter, type, timer=False, mininterval=120):
    assert type in ('process', 'thread', "sequential"), 'Type can be process, thread or sequential'
    
    if timer:
        t0 = time()

    if type == 'process':
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(func, my_iter), total=len(my_iter), mininterval=mininterval))
    elif type == 'thread':
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(func, my_iter), total=len(my_iter), mininterval=mininterval))
    elif type == "sequential":
        # Not parallel
        results = [func(item) for item in tqdm(my_iter)]

    if timer:
        print(f'{type} took {time()-t0} seconds')

    return results

def read_csv(input_file_path, delimiter=",", numeric=False):
    with open(input_file_path, "r") as f_in:
        reader = csv.DictReader(f_in, delimiter=delimiter)
        if numeric:
            data = [{key: float(value) for key, value in row.items()} for row in reader]
        else:
            data = [{key: value for key, value in row.items()} for row in reader]
    return data

def write_csv(data, output_file_path, append=False):
    fieldnames = list(data[0].keys())
    if append:
        with open(output_file_path, "a") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writerows(data)
    else:
        with open(output_file_path, "w") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

def json_load(input_file_path):
    with open(input_file_path, "r") as f_in:
        data = json.load(f_in)
    return data

def json_save(data, output_file_path):
    with open(output_file_path, "w") as f_out:
        json.dump(data, f_out, indent=2)


def find_common_nonzero_argmax(boxes1, boxes2):
    # Get a sample frame where there are prefferably faces and text
    # Find number of boxes per frame
    n_ocr = np.array([len(frame_boxes) if frame_boxes != None else 0 for frame_boxes in boxes1], dtype=float)
    n_face = np.array([len(frame_boxes) if frame_boxes != None else 0 for frame_boxes in boxes2], dtype=float)

    n_total = n_ocr + n_face

    # Replace 0 with minus infinity, so that we don't take those frames.
    n_ocr_inf = n_ocr.copy()
    n_ocr_inf[n_ocr==0] = -np.inf
    n_face_inf = n_face.copy()
    n_face_inf[n_face==0] = -np.inf

    n_total_inf = n_ocr_inf + n_face_inf

    # Find argmax (maximum number of boxes for face plus text)
    max_ind = np.argmax(n_total_inf)
    max_val = n_total_inf[max_ind]

    # If the maximum value is minus infinity, then get the maximum from original numbers.
    if max_val == -np.inf:
        max_ind = np.argmax(n_total)

    return max_ind


def convert_to_points(coords):
    # Converts points of corners to coordinates
    if len(coords) != 4:
        return []
    x1, y1, x2, y2 = coords
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def draw_boxes_on_image(image, shapes, labels=None, title=None):
    # To visualize the face emotion and OCR results
    if labels == None:
        labels = [None] * len(shapes)
    if labels and len(labels) != len(shapes):
        raise ValueError("Number of labels must match the number of shapes.")

    fig, ax = plt.subplots()
    ax.imshow(image)
    
    for i, shape in enumerate(shapes):
        if len(shape) != 4:
            raise ValueError("Each shape must contain exactly 4 points.")
        
        box_color = 'yellow'
        # If a label is provided, add it below each shape with lime color and black outline
        if labels and labels[i] is not None:
            box_color = 'lime'
            centroid_x = np.mean([point[0] for point in shape])
            centroid_y = np.max([point[1] for point in shape]) + 10  # Place label below the shape
            
            # Draw the text with a black outline
            text = ax.text(centroid_x, centroid_y, labels[i], fontsize=6, color='lime',
                           ha='center', va='top', fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                                   path_effects.Normal()])
            
        # Create a polygon patch for each shape with a yellow outline and black stroke
        polygon = patches.Polygon(shape, closed=True, edgecolor=box_color, linewidth=1, fill=None)
        polygon.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()])
        ax.add_patch(polygon)
    
    ax.axis('off')
    if title != None:
        ax.set_title(title, fontdict={'fontsize': 8})

    if get_backend() == 'agg':
        output_fp = 'sample_output_frame.png'
        plt.tight_layout()
        plt.savefig(output_fp)
        print(f'\nSample output frame saved to {output_fp}')
    else:
        plt.show(block=False)