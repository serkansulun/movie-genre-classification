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


def is_running_locally():
    username = getpass.getuser()
    return username == 'inesc'


def normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)

def standardize(tensor, mean_val, std_val):
    return (tensor - mean_val) / std_val

def get_tensor_bytes(x):
    return x.element_size() * x.numel()


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



# # # aucprc: from sklearn.metrics import roc_auc_score

# # def print_model(model):
# #     for name, layer in model.named_modules():
# #         print(name, layer)

# def tensor_show(x):
#     # plt.close()
#     x = (x - x.min()) / (x.max() - x.min())
#     x = x.squeeze()
#     if len(x.shape) == 3:
#         x = x.unsqueeze(0)  # add batch dimension
#     if x.shape[1] <=3:
#         # make it channel-last
#         if isinstance(x, np.ndarray):
#             x = x.transpose(0, 2, 3, 1)
#         else:
#             x = x.permute(0, 2, 3, 1)
#     for im in x:
#         plt.figure()
#         plt.imshow(im)
#     plt.show()

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
    # print(ret)
    return ret
        

# # def three_crop(img, size):
# #     if isinstance(size, int):
# #         size = (int(size), int(size))
# #     elif isinstance(size, (tuple, list)) and len(size) == 1:
# #         size = (size[0], size[0])

# #     if len(size) != 2:
# #         raise ValueError("Please provide only two dimensions (h, w) for size.")

# #     image_width, image_height = F.get_image_size(img)
# #     crop_height, crop_width = size
# #     if crop_width > image_width or crop_height > image_height:
# #         msg = "Requested crop size {} is bigger than input size {}"
# #         raise ValueError(msg.format(size, (image_height, image_width)))

# #     tl = F.crop(img, 0, 0, crop_height, crop_width)
# #     br = F.crop(img, 0, image_width - crop_width, crop_height, crop_width)

# #     center = F.center_crop(img, [crop_height, crop_width])

# #     return tl, center, br 


# # class ThreeOrSixCrop(torch.nn.Module):

# #     def __init__(self, size, hflip):
# #         super().__init__()
# #         self.size = size
# #         self.hflip = hflip

# #     def forward(self, img):
# #         first_three = three_crop(img, self.size)
# #         if self.hflip:
# #             img = F.hflip(img)
# #             second_three = three_crop(img, self.size)
# #             return first_three + second_three
# #         else:
# #             return first_three



# # def get_padding(image):
# #     max_w = 1203 
# #     max_h = 1479
    
# #     imsize = image.size
# #     h_padding = (max_w - imsize[0]) / 2
# #     v_padding = (max_h - imsize[1]) / 2
# #     l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
# #     t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
# #     r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
# #     b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    
# #     padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    
# #     return padding



# def imshow(x):
#     # Shows a single or batched image
#     plt.ion()
#     if torch.is_tensor(x):
#         if len(x.shape) == 2 or x.shape[0] == 3:   # single grayscale or color
#             x = x.unsqueeze(0)  # add batch dimension
#         for i in range(x.shape[0]):     # loop batch
#             im = x[i]
#             if len(im.shape) == 3:  # color
#                 im = im.permute(1, 2, 0)    # channel-last
#             plt.figure()
#             plt.imshow(im)
#     else:
#         plt.figure()
#         plt.imshow(x)
#     plt.show()

def unique_ordered_list(x):
    return list(dict.fromkeys(x))



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


def parse_debug():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args.debug

def flatten_list(x):
    return [item for sublist in x for item in sublist]

def write_class_performance(labels, class_performance, overall_performance, step, output_path, csv_also=False):
    # Ordering based on Movienet paper
    with open(output_path, "a") as f:
        f.write("Step: {:7d}\n".format(step))
        header = [""] + list(class_performance.keys())
        f.write("{:11s}  {:>6s}  {:>6s}  {:>6s}\n".format(*header))
        f.write("-" * 35 + "\n")
        for i, label in enumerate(labels):
            f.write("{:11s}  {:6.2f}  {:6.2f}  {:6.2f}\n".format(
                label,
                100 * abs(class_performance[header[1]][i]),
                100 * abs(class_performance[header[2]][i]),
                100 * abs(class_performance[header[3]][i])
            ))
        f.write("{:11s}  {:6.2f}  {:6.2f}  {:6.2f}\n".format(
                "MICRO AV.",
                100 * overall_performance["precision_micro"],
                100 * overall_performance["recall_micro"],
                100 * overall_performance["map_micro"],
            ))
        f.write("{:11s}  {:6.2f}  {:6.2f}  {:6.2f}\n".format(
                "MACRO AV.",
                100 * overall_performance["precision_macro"],
                100 * overall_performance["recall_macro"],
                100 * overall_performance["map_macro"],
            ))
        

        f.write("\n")
    if csv_also:
        output_path = output_path.replace(".txt", ".csv")
        # Create table (list of dicts)
        table = []
        for i, label in enumerate(labels):
            entry = {}
            entry[""] = label
            for key, value in class_performance.items():
                entry[key] = "{:.2f}".format(value[i]*100)   # round(value[i], 3)
            table.append(deepcopy(entry))

        entry = {}
        entry[""] = "MICRO AV."
        for key in class_performance.keys():
            if key[:2] == "P@":
                value = overall_performance["precision_micro"]
            if key[:2] == "R@":
                value = overall_performance["recall_micro"]
            if key[:2] == "AP":
                value = overall_performance["map_micro"]
            entry[key] = "{:.2f}".format(value*100) # round(value * 100, 2)
        table.append(deepcopy(entry))

        entry = {}
        entry[""] = "MACRO AV."
        for key in class_performance.keys():
            if key[:2] == "P@":
                value = overall_performance["precision_macro"]
            if key[:2] == "R@":
                value = overall_performance["recall_macro"]
            if key[:2] == "AP":
                value = overall_performance["map_macro"]
            entry[key] = "{:.2f}".format(value*100)
        table.append(deepcopy(entry))
        write_csv(table, output_path)


def scale(x, min_max=None):
    # Scales data between 0 and 1
    if min_max == None:
        min_ = x.min()
        max_ = x.max()
    else:
        min_ = min_max[0]
        max_ = min_max[1]
    return (x - min_) / (max_ - min_)


def plot_boxplots(data_dict, filename):
    # Extract keys and values from the dictionary
    labels = list(data_dict.keys())
    data = list(data_dict.values())
    
    # Create the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels,)
    
    # Set labels and title
    # plt.xlabel('Categories')
    # plt.ylabel('Values')
    # plt.title('Boxplots of Each Array')
    plt.grid()
    
    # Save the figure to the provided filename
    plt.savefig(filename)
    plt.close()


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
    # plt.ion()
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
        # shape = np.array(shape) + 0.2 * (np.array(shape) - np.mean(shape, axis=0))
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