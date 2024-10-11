from pathlib import Path
from preprocessing.src import feature_extractors
from classification.src.utils_classify import ind2multihot
from tqdm import tqdm
import utils as u
import torch
import numpy as np

debug = u.parse_debug()

if torch.cuda.is_available():
    device = 'cuda'  
else:
    device = 'cpu'
    print('--- USING CPU ---')

main_dir = Path('preprocessing/data')
print('remove later')
main_dir = Path("../../../../datasets/public/MovieNet")
trailers_dir = main_dir / 'trailers'
clean_labels_path = main_dir / "labels/trailers_genres_clean.csv"

extension = 'mkv'
use_scenecuts = True
fps = None

# Get labels
metadata = u.read_csv(clean_labels_path)
# convert from string to list
for i in range(len(metadata)):
    metadata[i]["genres"] = eval(metadata[i]["genres"])
metadata = {sample["youtube_id"] + '.mkv': sample["genres"] for sample in metadata}
valid_movienet_ids = sorted(list(metadata.keys()))

assert not (use_scenecuts and fps != None), 'Either use fps or scene cuts, not both.'

videos_dir = trailers_dir / 'downloaded'
video_paths = sorted(videos_dir.glob(f"**/*.{extension}"))
# video_paths = sorted(videos_dir.glob(f"**/*"))


# input_path = main_dir / 'features.pt'
output_path = main_dir / 'features.pkl'

# if debug:
#     input_path = trailers_dir / 'features_debug.pt'

# if input_path.exists():
#     x = torch.load(input_path)
# else:
#     print("Starting from scratch.")
#     x = {'samples': {}}     # will hold all the data

x = {'samples': {}}     # will hold all the data

features = ('asr_sentiment', 'face_emotion', 'ocr_sentiment', 'clip', 'beats', )
# features = ('beats', 'face_emotion',  )

for i, feature in enumerate(features):

    if feature == 'clip':
        model = feature_extractors.CLIPRunner()
    elif feature == 'beats':
        model = feature_extractors.BEATSRunner()
    elif feature == 'asr_sentiment':
        model = feature_extractors.ASRSentiment(tiny_asr=debug)
    elif feature == 'ocr_sentiment':
        model = feature_extractors.OCRPipeline()
    elif feature == 'face_emotion':
        model = feature_extractors.FaceExtractAndClassify()

    model.to_device(device)

    for video_path in tqdm(video_paths, mininterval=120, desc=f"{feature} - {len(features) - i} features remaining"):
        if True:
            video_name = video_path.stem
            x['samples'][video_name] = x['samples'].get(video_name, {})
            x['samples'][video_name]['features'] = x['samples'][video_name].get('features', {})
            # if feature not in x['samples'][video_name]['features'].keys():
            if True:
                video_output = model.process_video(video_path=video_path, fps=fps, use_scenecuts=use_scenecuts)
                video_feature = video_output['features']
                if not u.is_empty(video_feature) and video_feature is not None:
                    video_feature = u.detach_tensor(video_feature)
                x['samples'][video_name]['features'][feature] = video_feature

        # except Exception as e:
        #     print("Error", feature, video_path)
        #     print(e)
        
        if debug:
            break

print('Concatenating', flush=True)
# Calculate statistics for each feature
# Initialize a dictionary to hold the concatenated data for each feature
concatenated_data = {}

# Flatten and concatenate data from all samples for each feature
for sample in tqdm(x['samples'].values(), mininterval=120):
    for feature, tensor in sample['features'].items():
        if not u.is_empty(tensor):
            vector = tensor.flatten()
            if feature not in concatenated_data:
                concatenated_data[feature] = vector
            else:
                concatenated_data[feature] = np.concatenate((concatenated_data[feature], vector))

# Calculate the statistics for each feature
x['stats'] = {}
for feature, tensor in concatenated_data.items():
    x['stats'][feature] = {
        'mean': np.mean(tensor),
        'std': np.std(tensor),
        'min': np.min(tensor),
        'max': np.max(tensor)
    }
del concatenated_data

if not debug:
    torch.save(x, output_path)
    print('Saved to', output_path)
    torch.save(x['stats'], 'preprocessing/data/stats.pt')

print('Adding labels', flush=True)

all_genres = sorted(["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"])
genre_to_ind = {genre: ind for ind, genre in enumerate(sorted(all_genres))}
ind_to_genre = {ind: genre for ind, genre in enumerate(sorted(all_genres))}

x['idx_to_label'] = ind_to_genre

to_delete = []
for video in tqdm(x['samples'].keys(), mininterval=120):
    if video in metadata.keys():
        genres = metadata[video]
        inds = [genre_to_ind[genre] for genre in genres]
        multihot = ind2multihot(inds, len(all_genres))
        x['samples'][video]['label'] = u.detach_tensor(multihot)
        # x['samples'][video]['video_path'] = video
    else:
        to_delete.append(video)
for video in to_delete:
    del x['samples'][video]
        
if not debug:
    u.pickle_save(x, output_path)
    print('Saved to', output_path)



