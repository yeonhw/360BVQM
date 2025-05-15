import os
import sys
import json
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter, writer
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
from opts import parse_opts
from model.network import C3DVQANet
from dataset.dataset import VideoDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda'


def train_model(model, device, optimizer, scheduler, dataloaders, save_checkpoint, epoch_resume=1, num_epochs=25):

    for epoch in tqdm(range(epoch_resume, num_epochs+epoch_resume), unit='epoch', initial=epoch_resume, total=num_epochs+epoch_resume):
        print('epoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            epoch_labels = []
            epoch_preds = []
            feat_map = []
            epoch_loss = 0.0
            epoch_size = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for ref, dis, fast_feature, erp_glob, labels in dataloaders[phase]:
                ref = ref.to(device)
                dis = dis.to(device)
                erp_glob = erp_glob.to(device)
                fast_feature = fast_feature.to(device)
                labels = labels.to(device).float()

                ref = ref.reshape(-1, ref.shape[2], ref.shape[3], ref.shape[4], ref.shape[5])
                dis = dis.reshape(-1, dis.shape[2], dis.shape[3], dis.shape[4], dis.shape[5])

                erp_glob = erp_glob.reshape(erp_glob.shape[1], erp_glob.shape[2], erp_glob.shape[3], erp_glob.shape[4])
                fast_feature = fast_feature.reshape(fast_feature.shape[1], fast_feature.shape[2])

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(ref, dis, fast_feature, erp_glob)
                    preds = torch.mean(preds, 0, keepdim=True)
                    preds = preds.reshape(1)
                    criterion = nn.MSELoss()
                    loss = criterion(preds, labels)

                    if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
                        loss = torch.mean(loss)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * labels.size(0)
                epoch_size += labels.size(0)
                epoch_labels.append(labels.flatten())
                epoch_preds.append(preds.flatten())

            epoch_loss = epoch_loss / epoch_size

            if phase == 'train':
                scheduler.step(epoch_loss)

            epoch_labels = torch.cat(epoch_labels).flatten().data.cpu().numpy()
            epoch_preds = torch.cat(epoch_preds).flatten().data.cpu().numpy()

            logging.info('epoch_labels: {}'.format(epoch_labels))
            logging.info('epoch_preds: {}'.format(epoch_preds))

            epoch_plcc = pearsonr(epoch_labels, epoch_preds)[0]
            epoch_srocc = spearmanr(epoch_labels, epoch_preds)[0]
            epoch_krocc = kendalltau(epoch_labels, epoch_preds)[0]
            epoch_rmse = np.sqrt(np.mean((epoch_labels - epoch_preds)**2))

            logging.info("{phase}-Loss: {loss:.4f}\t RMSE: {rmse:.4f}\t PLCC: {plcc:.4f}\t SROCC: {srocc:.4f}\t KROCC: {krocc:.4f}".format(phase=phase, loss=epoch_loss, rmse=epoch_rmse, plcc=epoch_plcc, srocc=epoch_srocc, krocc=epoch_krocc))
            print("{phase}-Loss: {loss:.4f}\t RMSE: {rmse:.4f}\t PLCC: {plcc:.4f}\t SROCC: {srocc:.4f}\t KROCC: {krocc:.4f}".format(phase=phase, loss=epoch_loss, rmse=epoch_rmse, plcc=epoch_plcc, srocc=epoch_srocc, krocc=epoch_krocc))

            if phase == 'test' and save_checkpoint:
                _checkpoint = '{pt}_{epoch}'.format(pt=save_checkpoint, epoch=epoch)
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, _checkpoint)#单gpu加载需要去除model后的module


if __name__=='__main__':

    opt = parse_opts()

    video_path = opt.video_dir
    subj_dataset = opt.score_file_path
    save_checkpoint = opt.save_model
    load_checkpoint = opt.load_model
    log_file_name = opt.log_file_name
    LEARNING_RATE = opt.learning_rate
    L2_REGULARIZATION = opt.weight_decay
    NUM_EPOCHS = opt.epochs
    MULTI_GPU_MODE = opt.multi_gpu
    channel = opt.channel
    size_x = opt.size_x
    size_y = opt.size_y
    stride_x = opt.stride_x
    stride_y = opt.stride_y

    logging.basicConfig(filename=log_file_name, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info('OK parse options')

    logging.info("Parsed arguments:")
    for arg, value in vars(opt).items():
        logging.info(f"{arg}: {value}")

    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y) for x in ['train', 'test']} # 'train',
    dataloaders = {x: torch.utils.data.DataLoader(video_dataset[x], batch_size=1, shuffle=True, num_workers=4, drop_last=True) for x in ['train', 'test']} #num_workers=8 'train',

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
        device_ids = range(0, torch.cuda.device_count())
        model = torch.nn.DataParallel(C3DVQANet().to(device), device_ids=device_ids)
        logging.info("muti-gpu mode enabled, use {0:d} gpus".format(torch.cuda.device_count()))
    else:
        model = C3DVQANet().to(device)

        logging.info('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)
    epoch_resume = 1

    if os.path.exists(load_checkpoint):
        checkpoint = torch.load(load_checkpoint)
        weights_dict = torch.load(load_checkpoint)["model_state_dict"]
        logging.info("loading checkpoint")

        if torch.cuda.device_count() > 1 and MULTI_GPU_MODE==True:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(model.load_state_dict(weights_dict, strict=False))

        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

    train_model(model, device, optimizer, scheduler, dataloaders, save_checkpoint, epoch_resume, num_epochs=NUM_EPOCHS)
