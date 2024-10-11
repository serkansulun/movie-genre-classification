from pathlib import Path
from classification.src.classifiers import init_model
import torch
import utils as u
import numpy as np
from tqdm import tqdm
from preprocessing.src import feature_extractors
import skvideo.io
import pandas as pd
from preprocessing.src import video_utils as u_video
import argparse

extractor = feature_extractors.BEATSRunner(predict=True)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--model_dir', type=str, help='Path to the model directory',
                    default='classification/output/single_transformer')
parser.add_argument('--video_path', type=str, help='Path to the video file',
                    # default='preprocessing/data/trailers/downloaded/3VDfF2Mxv0g.mkv',
                    default=None,
                    )
parser.add_argument('--youtube_link', type=str, required=False, help='Path to the video file',
                    # default='https://www.youtube.com/watch?v=ohF5ZO_zOYU',
                    default=None,
                    )

args = parser.parse_args()

assert bool(args.video_path) != bool(args.youtube_link), 'Provide either video path or YouTube link, and not both.'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if bool(args.video_path):
    video_path = Path(args.video_path)
else:
    target_dir = Path('preprocessing/data/youtube')
    target_dir.mkdir(exist_ok=True)
    video_path = Path(u_video.download_youtube(args.youtube_link, target_dir=target_dir))
    print(video_path)

model_dir = Path(args.model_dir)

groundtruth = pd.read_csv('preprocessing/data/labels/trailers_genres_clean.csv')

video_name = video_path.stem
if video_name in groundtruth['youtube_id'].values:
    groundtruth = eval(groundtruth.loc[groundtruth['youtube_id'] == video_name, 'genres'].values.item())
else:
    groundtruth = None

visualize = True
feed_tensor = True

use_scenecuts = True
fps = 1

labels = sorted(["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"])

config = torch.load(model_dir / 'config.pt')
config['device'] = device

feature_names = config['features']

stats = torch.load('preprocessing/data/stats.pt')

if use_scenecuts:
    input_frames, _ = u_video.video_to_midscenes(video_path)  # Get scenes  
else:
    input_frames = skvideo.io.vread(str(video_path))
        
    if fps != None:
        input_fps = u_video.get_video_fps(str(video_path))
        multiplier = input_fps / fps
        indices = np.round(np.arange(0, input_frames.shape[0] - 2, multiplier)).astype(np.int32).tolist()
        input_frames = input_frames[indices, ...]

# Get audio from video
input_audio, sr = u_video.extract_audio(video_path)

sample_features = {}
video_outputs = {}
clip_features = None    # We will use it later for captioning

with torch.no_grad():
    # PART I - FEATURE EXTRACTION    
    for feature_name in tqdm(feature_names, desc='Extracting features'):

        print(f'\nFeature: {feature_name}')
        if feature_name == 'clip':
            extractor = feature_extractors.CLIPRunner()
            input_tensor = input_frames
        elif feature_name == 'beats':
            extractor = feature_extractors.BEATSRunner(predict=True)
            input_tensor = input_audio
        elif feature_name == 'asr_sentiment':
            extractor = feature_extractors.ASRSentiment()
            input_tensor = input_audio
        elif feature_name == 'ocr_sentiment':
            extractor = feature_extractors.OCRPipeline()
            input_tensor = input_frames
        elif feature_name == 'face_emotion':
            extractor = feature_extractors.FaceExtractAndClassify() 
            input_tensor = input_frames

        extractor.to_device(device)

        if feed_tensor:
            video_output = extractor.process_video(input_tensor=input_tensor, sr=sr)
        else:
            video_output = extractor.process_video(video_path=video_path, n_frames=None, fps=1)

        extracted_feature = video_output['features']

        if feature_name == 'clip':
            clip_features = extracted_feature

        # Keep other output for visualization
        video_outputs[feature_name] = video_output

        if extracted_feature == []:     # Feature unavailable, use zeros
            extracted_feature = torch.zeros((config['feature_lengths'][feature_name],
                config['feature_dims'][feature_name]))
        else:
            if config['normalize']:     # min-max normalization
                extracted_feature = u.normalize(extracted_feature, stats[feature_name]['min'], stats[feature_name]['max'])
            if config['standardize']:   # standardize to zero mean unit variance
                extracted_feature = u.standardize(extracted_feature, stats[feature_name]['mean'], stats[feature_name]['std'])

        # Fix lengths
        source_length = extracted_feature.shape[0]
        target_length = config['feature_lengths'][feature_name]

        if source_length > target_length:     # Take equidistant frames
            inds = u.equidistant_indices(source_length, target_length)
            extracted_feature = extracted_feature[inds, :]
        elif source_length < target_length:
            extracted_feature = torch.nn.functional.pad(extracted_feature, (0, 0, 0, target_length - source_length))

        # Add batch dimension and move to device
        extracted_feature = extracted_feature.unsqueeze(0).to(device)

        sample_features[feature_name] = extracted_feature

    if visualize:   # Caption model is only used for visualization
        caption_model = feature_extractors.CaptionRunner()

    del extractor
    model = init_model(config)
    model.load_state_dict(torch.load(model_dir / 'model.pt', map_location=lambda storage, loc: storage))
    model.eval()

    # PART II - RUN MODEL
    with torch.cuda.amp.autocast(enabled=config['amp']):
        output = model(sample_features).squeeze()
        output = torch.nn.functional.softmax(output)
        output = u.detach_tensor(output)

# PART III - VISUALIZATION (OPTIONAL)
if visualize:

    # *** Optical Character Recognition (OCR) analysis
    if 'ocr_sentiment' in video_outputs.keys():
        ocr_text = video_outputs['ocr_sentiment']['ocr_processed']
        ocr_boxes = video_outputs['ocr_sentiment']['coordinates']
        if ocr_text != []:
            ocr_text = '. '.join(video_outputs['ocr_sentiment']['ocr_processed'])
            sentiment_prediction = video_outputs['ocr_sentiment']['predictions'][0].capitalize()
            sentiment_percentage = round(video_outputs['ocr_sentiment']['predictions'][1] * 100)
            print('\nOptical character recognition (OCR):')
            print(ocr_text)
            print('Sentiment analysis:')
            print(f"{sentiment_percentage}%  {sentiment_prediction}")
    else:
        print("OCR sentiment isn't in model output.")
        ocr_boxes = [None]

    if 'asr_sentiment' in video_outputs.keys():
        # *** Automatic Speech Recognition (ASR) analysis
        asr_text = video_outputs['asr_sentiment']['asr']
        asr_language = video_outputs['asr_sentiment']['language']
        if asr_text != "":
            sentiment_prediction = video_outputs['asr_sentiment']['predictions'][0].capitalize()
            sentiment_percentage = round(video_outputs['asr_sentiment']['predictions'][1] * 100)
            print('\nAutomatic speech recognition (ASR):')
            if asr_language not in ('english', None):
                print(f'(Translated from {asr_language.capitalize()}.)')
            print(asr_text)
            print('Sentiment analysis:')
            print(f"{sentiment_percentage}%  {sentiment_prediction}")
    else:
        print("ASR sentiment isn't in model output.")

    if 'beats' in video_outputs.keys():
        # *** BEATS classification
        # Using the entire video, print BEATS (audio event) probabilities
        predictions = video_outputs['beats']['predictions']
        if predictions != []:
            print('\nAudio classification:')
            for prediction in predictions:
                print(f'{round(prediction[1] * 100):2}%  {prediction[0]}')
    else:
        print("BEATS isn't in model output.")

    # *** FACE and OCR
    # Display a sample frame with OCR and facial boxes
    
    if 'face_emotion' in video_outputs.keys():
        face_boxes = video_outputs['face_emotion']['coordinates']
        face_predictions = video_outputs['face_emotion']['predictions']
    else:
        face_boxes = [None]
        face_predictions = [None]
        print("Face emotion isn't in model output.")

    # Find the frame with faces and text
    i = u.find_common_nonzero_argmax(ocr_boxes, face_boxes)

    ocr_boxes_frame = []
    if ocr_boxes != [None]:
        if ocr_boxes[i] != None:
            # Get OCR boxes for that frame    
            ocr_boxes_frame = [pred[0] for pred in ocr_boxes[i]]

    # We don't need predictions per frame, we will print OCR from the entire video
    ocr_predictions_frame = [None] * len(ocr_boxes_frame)

    face_boxes_frame = []
    face_predictions_frame = []
    if face_boxes != [None]:
        if face_boxes[i] != None:
            face_boxes_frame = face_boxes[i]
            face_boxes_frame = [u.convert_to_points(box) for box in face_boxes_frame]

            face_predictions_frame = face_predictions[i]
            
    face_predictions_frame = [f'{round(pred[1] * 100):2}%  {pred[0].upper()}' for pred in face_predictions_frame]

    boxes = face_boxes_frame + ocr_boxes_frame
    box_labels = face_predictions_frame + ocr_predictions_frame

    # *** CLIP caption
    # Using the same sample frame, create a caption using CLIP
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if clip_features == None:
        title = None
    else:
        clip_feature = clip_features[i:i+1, ...].to(device)
        with torch.no_grad():
            caption = caption_model(clip_feature)
            title = f'Predicted caption: {caption.capitalize()}'

    selected_frame = input_frames[i]

    u.draw_boxes_on_image(selected_frame, boxes, labels=box_labels, title=title)
    
# Print emotion prediction results
sorted_indices = np.argsort(output)[::-1]

print('\nPrediction:')
for idx in sorted_indices[:5]:
    print(f"{round(output[idx] * 100, 1):5}%  {labels[idx]}")

if groundtruth != None:
    print('\nGround-truth:\n' + '\n'.join(groundtruth))


