import os
import argparse
import datetime
from pathlib import Path
import torch
import utils as u
# from random import randint

parser = argparse.ArgumentParser(description='Video emotion classification')

parser.add_argument('--note', type=str, default='None',
                    help='Add a note')
parser.add_argument('--model', default="single_transformer", 
                    choices=['multi_transformer', 'single_transformer', 'mlp'], help='Type of model')
parser.add_argument('--dataset_dir', 
                    default="preprocessing/data", help='Directory of the dataset')
parser.add_argument("--features", nargs='+', default=None,
                    help="Features to use")
parser.add_argument('--n_frames', type=int, default=None,
                    help='Number of frames to use for each feature')
parser.add_argument('--restart_dir', type=str, default=None,
                    help='Directory to load trained model')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='Learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate')
parser.add_argument('--n_layers', default=1, type=int,
                    help='Number of hidden layers')
parser.add_argument('--d_model', default=1280, type=int,
                    help='Size of model layers. -1 for no encoding and using features as is.')
parser.add_argument('--n_heads', default=8, type=int,
                    help='Number of transformer heads')
parser.add_argument('--pos_weight', type=float, default=-1,
                    help='Loss weight for positive samples')
parser.add_argument('--log_step', type=int, default=25000,
                    help='report interval')
parser.add_argument('--eval_step', type=int, default=25000,
                    help='evaluation interval')
parser.add_argument('--skip_first_eval', action='store_true',
                    help='Skips the evaluation before training')
parser.add_argument('--debug', action='store_true',
                    help='Debug mode, doesnt create files')
parser.add_argument('--test_only', action='store_true',
                    help='Only tests on test split, no training')
parser.add_argument('--trn_val_tst_ratio', type=float, default=[0.7, 0.1, 0.2],
                    help='Train split ratio')
parser.add_argument('--normalize', action='store_true',
                    help='Normalize data')
parser.add_argument('--standardize', action='store_true',
                    help='Standardize data')
parser.add_argument('--no_amp', action="store_true",
                    help='Disable automatic mixed precision')
parser.add_argument('--no_cuda', action='store_true',
                    help='use CPU')
parser.add_argument('--pin_memory', action='store_true',
                    help='Pin memory')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of cores for data loading')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed')
parser.add_argument('--reset_scaler', action="store_true",
                    help="Reset scaler (can help avoiding nans)")
parser.add_argument("--overwrite_lr", action="store_true", 
                    help="Overwrites learning rate if pretrained model is loaded.")
parser.add_argument('--accumulate_step', type=int, default=1,
                    help='Accumulate gradients (multiplies effective batch size')
parser.add_argument("--overwrite_dropout", action="store_true",
                    help="Resets dropouts")
parser.add_argument('--max_eval_step', type=int, default=-1,
                    help='maximum evaluation steps')
parser.add_argument('--max_step', type=int, default=1000000000,
                    help='maximum training steps')
parser.add_argument('--patience', type=int, default=100000,
                    help='If validation performance is not improving after n steps, stop training')
parser.add_argument("--fixed_position_encoding", action="store_true",
                    help="Use fixed position encoding rather than learned position")
args = parser.parse_args()

args.amp = not args.no_amp


# Data

'''
Upper (non-extreme) ranges -> Multiple of 2:

BEATS: 680 -> 704
CLIP:  357 -> 384
Face:  622 -> 640
ASR:   61  ->  64

Total: 1792 (divisible by 256)
'''

# Upper whiskers are in comments (Q3 * 1.5 (Q3 - Q1))
# The lengths are sum of large powers are two,
# that are slightly larger than the upper whiskers
if args.n_frames == None or args.n_frames <= 0:
    args.feature_lengths = {
            'asr_sentiment': 1,
            'beats': 144,   # 137.5
            'clip': 160,    # 150.5
            'face_emotion': 80,    # 78.0
            'ocr_sentiment': 1,
        }
else:
    args.feature_lengths = {
            'asr_sentiment': 1,
            'beats': args.n_frames,   # 137.5
            'clip': args.n_frames,    # 150.5
            'face_emotion': args.n_frames,    # 78.0
            'ocr_sentiment': 1,
        }

if args.model == 'multi_transformer':
    for key in args.feature_lengths:
        if 'sentiment' not in key:
            args.feature_lengths[key] -= 1      # For <CLS> token
    
args.feature_dims = {
    'asr_sentiment': 768,
    'beats': 768,
    'clip': 512,
    'face_emotion': 768,
    'ocr_sentiment': 768,
}

args.all_features = sorted(list(args.feature_lengths.keys()))
if args.features == None:
    args.features = args.all_features
else:
    args.feature_lengths = {key: value for key, value in args.feature_lengths.items() if key in args.features}
    args.feature_dims = {key: value for key, value in args.feature_dims.items() if key in args.features}

args.labels = sorted(["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"])

args.n_labels = len(args.labels)

if args.model == 'mlp':
    assert args.d_model != None, "MLP requires d_model to be specified"

# Device
use_cuda = torch.cuda.is_available() and not args.no_cuda
args.device = torch.device('cuda' if use_cuda else 'cpu')

# Output directories and files
args.main_output_dir = Path("classification/output")
args.start_time = datetime.datetime.now()
if not u.is_running_locally():
    args.start_time += datetime.timedelta(hours=1)
args.start_time = args.start_time.strftime("%Y_%m_%d_%H_%M_%S_%f")
args.output_dir = args.main_output_dir / args.start_time

if args.restart_dir != None:
    args.restart_dir = args.main_output_dir / args.restart_dir

if args.test_only:
    assert args.restart_dir != None, "Need trained model (--restart_dir) for testing"
