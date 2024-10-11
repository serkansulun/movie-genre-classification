# Movie trailer (video) genre classification using multimodal pretrained features

Serkan Sulun - 2024

## Citation

If you use our code in a work that leads to a scientific publication, we would appreciate it if you would kindly cite our paper in your manuscript: `https://doi.org/10.1016/j.eswa.2024.125209`

## Install requirements:

If you have Conda (optional):
```
conda create -n genre python=3.12
conda activate genre
```
Then
```
pip install -r requirements
```


## If you want to run a pretrained classifier on a single video (inference):

1- Download pretrained models

```
python -m classification.src.download_pretrained_models
```

2- Run model on a video

```
python -m classification.src.inference --model_dir classification/output/multi_transformer --video_path preprocessing/data/trailers/downloaded/3VDfF2Mxv0g.mkv
```

Change the arguments for `--model_dir` and `--video_path` as you'd like.

## If you want to train your own model:

1- Download extracted features

```
python -m classification.src.download_features
```

2- Train (see `classification/src/config.py` for arguments)

```
python -m classification.src.train
```

## If you want to download MovieNet videos and extract the features yourself:

1- Download MovieNet:

```
python -m preprocessing.src.download_movienet
```

2- Extract pretrained features

```
python -m preprocessing.src.extract_features
```

## features.pkl explained

You can open the `features.pkl` file using the `Pickle` library. It results in a dictionary with the following keys:

`samples`: This yields another dictionary where the keys are the YouTube IDs, corresponding to the videos. The values are dictionaries with the following keys. `labels` contain multi-hot labels. `features` contain tensors belonging to each type of feature.

`stats`: Statistics (minimum, maximum, mean, standard deviation) for each type of feature. This is useful for normalization or standardization.

`idx_to_label`: Dictionary which maps indices to categories of cinematic genres. This helps making sense of the multi-hot labels.

## splits.json explained

Contains the training, validation and testing splits for the MovieNet dataset.

To create it, we sorted the YouTube IDs of the trailers alphabetically, and sliced them using percentages of 70%, 10%, and 20%, for training, validation, and testing, respectfully. The original MovieNet paper (Huang et al., 2020) also uses these percentages.

You should never use or view your testing split until reporting final results. For hyperparameter optimization or early stopping, you should always use the validation split. Once you have your final model trained, you should perform inference on the testing split only once, and report those results. This is the ethical way of doing machine learning research. This is necessary for two reasons. First, the trained models should work on unseen real-world scenarios. The unseen testing split would replicate this scenario. Second, it provides a fair competition because one can tune the hyperparemeters to optimize performance on - or even overfit - the testing split, or exploit a lucky random initialization to report superior results on the testing split.



