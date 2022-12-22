"""Model training script"""

import re, random, math
import time
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import load_datasets
from faceformer_video_reconstruction import Faceformer

def trainer(args, train_loader, val_loader, test_loader, model, optimizer, vertice_criterion, video_criterion, video_lambda, epoch=100, to_cuda:bool =False):
    """Model training loop:
    
    Args:
        - args: command line arguments as outlined in main()
        - train_loader: data loader for training set
        - val_loader: data loader for validation ste
        - test_loader: data loader for test set
        - model: model
        - optimizer: optimizer
        - vertice_criterion: loss function for evaluating vertice predictions
        - video_criterion: loss function for evaluating video reconstruction
        - video_lambda: tuning parameter to weight video loss
        - epoch: number of epochs
        - to_cuda: T/F representing whether to train on CUDA or not
    """
    save_path = Path(args.dataset) / Path(args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    for e in range(epoch):
        train_running_loss = 0
        model.train()
        loader = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()

        # training loop
        for i, (audio, vertice, template, one_hot, video, file_name) in loader:
            if to_cuda:     # to gpu
                audio, vertice, template, one_hot, video  = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda"), video.to(device="cuda")

            vertice_out, video_out = model(audio, template,  vertice, one_hot, video)

            loss = vertice_criterion(vertice_out, vertice) + video_lambda * video_criterion(video_out, video)
            loss = torch.mean(loss)
            loss.backward()

            train_running_loss += loss.item()

            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()

            loader.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), i+1 , train_running_loss / (i+1)))

        
        val_running_loss = 0
        model.eval()

        # validation loss
        for i, (audio, vertice, template, one_hot_all, video, file_name) in enumerate(val_loader):
            if to_cuda:
                audio, vertice, template, one_hot_all, video= audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda"), video.to(device="cuda")

            train_subject = "_".join(file_name[0].split("_")[:-1])

            if train_subject in train_subjects_list:
                idx = train_subjects_list.index(train_subject)
                one_hot = one_hot_all[:,idx,:]
                vertice_out, video_out = model(audio, template, vertice, one_hot, video)

                loss = vertice_criterion(vertice_out, vertice) + (video_lambda * video_criterion(video_out, video) / 4)
                loss = torch.mean(loss)
                val_running_loss += loss.item()

            else:
                for idx in range(one_hot_all.shape[-1]):
                    one_hot = one_hot_all[:,idx,:]
                    vertice_out, video_out = model(audio, template, vertice, one_hot, video)
                    loss = vertice_criterion(vertice_out, vertice) + video_lambda * video_criterion(video_out, video)
                    loss = torch.mean(loss)
                    val_running_loss += loss.item()
                        
        val_loss = val_running_loss / (i+1)
        
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

        if ((e+1) % 4 == 0):
            test(args, model, test_loader, epoch=args.max_epoch, to_cuda=to_cuda)

        print("epoch: {}, validation loss:{:.7f}".format(e+1, val_loss))

    return model


@torch.no_grad()
def test(args, model, test_loader, epoch, to_cuda=False):
    """Model testing loop. Save test predictions to disk."""
    result_path = os.path.join(args.dataset,args.result_path)

    save_path = os.path.join(args.dataset,args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))

    if to_cuda:
        model = model.to(torch.device("cuda"))

    model.eval()
    ttt = time.time()
    loss = nn.MSELoss()
    running_loss = 0

    for i, (audio, vertice, template, one_hot_all, video, file_name) in enumerate(test_loader):
        if to_cuda:     # to gpu
            audio, vertice, template, one_hot_all, video = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda"), video.to(device="cuda")

        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            idx = train_subjects_list.index(train_subject)
            one_hot = one_hot_all[:,idx,:]
            prediction = model.predict(audio, template, one_hot)

            l = loss(vertice, prediction)
            l = torch.mean(l)
            running_loss += l

            prediction = prediction.squeeze()
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
            

        else:
            for idx in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[idx]
                one_hot = one_hot_all[:,idx,:]
                prediction = model.predict(audio, template, one_hot)

                l = loss(prediction, vertice)
                l = torch.mean(l)
                running_loss += l

                prediction = prediction.squeeze()
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())

        
    print("loss:", running_loss / (i+1))
    print("test time:", time.time() - ttt)
         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="vocaset", help='path to dataset')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--videos_path", type=str, default="videos_npy", help="path of the videos")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=25, help='number of epochs')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA"
       " FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA"
       " FaceTalk_170731_00024_TA")
    args = parser.parse_args()

    #build model
    model = Faceformer(args)
    print("model parameters: ", count_parameters(model))

    to_cuda = torch.cuda.is_available()

    if to_cuda:
        model = model.to(torch.device('cuda'))
    
    #load data
    dataset_root = Path(args.dataset)
    audio_path = dataset_root / Path(args.wav_path)
    vertice_path = dataset_root / Path(args.vertices_path)
    videos_path = dataset_root / Path(args.videos_path)

    train_dataset, valid_dataset, test_dataset = load_datasets(audio_path, vertice_path, videos_path, args.train_subjects, args.val_subjects, args.test_subjects)

    # loss
    vertice_criterion = nn.MSELoss()
    video_criterion = nn.MSELoss()

    # Train the model
    video_lambda = (1e-10)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    model = trainer(args, train_dataset, valid_dataset, test_dataset, model, optimizer, vertice_criterion, video_criterion, video_lambda, epoch=args.max_epoch, to_cuda=to_cuda)

    # test model
    # test(args, model, valid_dataset, epoch=args.max_epoch, to_cuda=to_cuda)   

    
if __name__=="__main__":
    main()