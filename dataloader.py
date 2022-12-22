"""
Custom data loader for training using VOCA. Reads audio files into memory but loads in vertice
and video data on each call to __getitem__.
"""

import cv2
import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    


def video_to_npy(video_path: Path) -> np.ndarray:
    """Given a path to a video (mp4), return the video as a np.array of images"""
    video = np.load(video_path, allow_pickle=True)
    video = np.moveaxis(video,-1, 1)
    
    return video


def vertices_to_npy(vertice_path: Path) -> np.ndarray:
    """Given a path to vertices, return them as a numpy array"""
    return np.load(vertice_path, allow_pickle=True)[::2,:]


class Dataset(data.Dataset):
    """Custom dataloader"""

    def __init__(self, data, subjects, processor, split: str):
        """Initialize dataloader
        
        Args:
            - data: list of examples
            - subjects: list of speaker names
            - processor: audio processor
            - split: dataset split
        """
        self.data = data
        self.len = len(self.data)
        self.subjects = subjects
        self.processor = processor
        self.split = split
        self.one_hot_vec = np.eye(len(subjects["train"]))

    def __getitem__(self, index):
        """Return one example. Audio is already loaded into memory, but
        video and vertice data is read into memory from storage.
        """
        file_name = self.data[index]["name"]
        template = self.data[index]["template"]

        # get audio
        audio = self.data[index]["audio"]
        
        # get vertice
        vertice_file = self.data[index]["vertice_file"]
        vertice = vertices_to_npy(vertice_file)

        # get video
        video_file = self.data[index]["video_file"]
        video = video_to_npy(video_file)

        # get one hot encoding
        if self.split == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_vec[self.subjects["train"].index(subject)]
        else:
            one_hot = self.one_hot_vec

        return torch.FloatTensor(audio), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), torch.FloatTensor(video), file_name

    def __len__(self):
        return self.len


def read_data(audio_path, vertices_path, video_path, train_subjects, val_subjects, test_subjects):
    print("Loading data...")
    data = defaultdict(dict)

    template_file = "vocaset/templates.pkl"
    with open(template_file, "rb") as f:
        templates = pickle.load(f, encoding="latin1")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    for audio_file in audio_path.glob("*.wav"):
        key = str(audio_file.stem)
        subject_id = "_".join(key.split("_")[:-1])

        vertice_file = vertices_path / Path(key + ".npy")
        video_file = video_path / Path(key + ".npy")

        if not vertice_file.exists() or not video_file.exists() or not audio_file.exists():
            continue
        
        audio_array, sample_rate = librosa.load(audio_file, sr=16000)
        audio = np.squeeze(processor(audio_array,sampling_rate=16000).input_values)

        data[key]["audio"] = audio
        data[key]["vertice_file"] = str(vertice_file)
        data[key]["video_file"] = str(video_file)

        data[key]["name"] = key
        data[key]["template"] = templates[subject_id].reshape((-1))


    subjects_dict = {}
    subjects_dict["train"] = [subject for subject in train_subjects.split(" ")]
    subjects_dict["val"] = [subject for subject in val_subjects.split(" ")]
    subjects_dict["test"] = [subject for subject in test_subjects.split(" ")]

    splits = {"train": range(1,41), "val": range(21,41), "test": range(21,41)}

    train_data = []
    valid_data = []
    test_data = []

    for key, value in data.items():
        subject_id = "_".join(key.split("_")[:-1])
        sentence_id = int(key.split(".")[0][-2:])

        if subject_id in subjects_dict["train"] and sentence_id in splits['train']:
            train_data.append(value)
        elif subject_id in subjects_dict["val"] and sentence_id in splits['val']:
            valid_data.append(value)
        elif subject_id in subjects_dict["test"] and sentence_id in splits['test']:
            test_data.append(value)

    return processor, train_data, valid_data, test_data, subjects_dict


def load_datasets(audio_path, vertice_path, video_path, train_subjects, val_subjects, test_subjects):
    processor, train_data, valid_data, test_data, subjects_dict = read_data(audio_path, vertice_path, video_path, train_subjects, val_subjects, test_subjects)

    train_dataset = Dataset(train_data, subjects_dict, processor, "train")
    validation_dataset = Dataset(valid_data, subjects_dict, processor, "val")
    test_dataset = Dataset(test_data, subjects_dict, processor, "test")

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    valid_loader = data.DataLoader(dataset=validation_dataset, batch_size=1, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    return train_loader, valid_loader, test_loader