# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:51:44 2021

@author: DELL
"""
import glob
import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
import time
import torch.utils.data as D
from torchvision import transforms as T

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

class TestDataset(D.Dataset):
    def __init__(self, pca_image_paths, seg_pca_image_paths):
        self.pca_image_paths = pca_image_paths
        self.seg_pca_image_paths = seg_pca_image_paths
        self.len = len(pca_image_paths)
        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        pca_image = cv2.imread(self.pca_image_paths[index],cv2.IMREAD_UNCHANGED)
        pca_image = cv2.resize(pca_image,(512,512))
        
        seg_pca_image = cv2.imread(self.seg_pca_image_paths[index],cv2.IMREAD_UNCHANGED)
        seg_pca_image = cv2.resize(seg_pca_image,(512,512))
        return self.as_tensor(pca_image), self.as_tensor(seg_pca_image), self.pca_image_paths[index]
    # 数据集数量
    def __len__(self):
        return self.len

def get_testdataloader(pca_image_paths, seg_pca_image_paths, batch_size, 
                       shuffle, num_workers):
    dataset = TestDataset(pca_image_paths, seg_pca_image_paths)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

def test(pca_model_paths, seg_pca_model_paths, output_dir, test_loader):
    in_channels = 4
    model = smp.UnetPlusPlus(
        encoder_name="timm-resnest101e",
        encoder_weights="imagenet",
        decoder_attention_type="scse",
        in_channels=in_channels,
        classes=5,
        )
    
    for pca_image, seg_pca_image, pca_image_paths in test_loader:
        
        model.to(DEVICE)
        output = np.zeros((1,5,512,512))
        for pca_model_path in pca_model_paths:
            model.load_state_dict(torch.load(pca_model_path))
            model.eval()
            with torch.no_grad():
                # image.shape: 16,4,256,256
                pca_image_flip2 = torch.flip(pca_image,[2])
                pca_image_flip2 = pca_image_flip2.cuda()
                pca_image_flip3 = torch.flip(pca_image,[3])
                pca_image_flip3 = pca_image_flip3.cuda()
                
                pca_image = pca_image.cuda()
                output1 = model(pca_image).cpu().data.numpy() 
                output2 = torch.flip(model(pca_image_flip2),[2]).cpu().data.numpy()
                output3 = torch.flip(model(pca_image_flip3),[3]).cpu().data.numpy()
                
            output += output1 + output2 + output3
        
        for seg_pca_model_path in seg_pca_model_paths:
            model.load_state_dict(torch.load(seg_pca_model_path))
            model.eval()
            with torch.no_grad():
                # image.shape: 16,4,256,256
                seg_pca_image_flip2 = torch.flip(seg_pca_image,[2])
                seg_pca_image_flip2 = seg_pca_image_flip2.cuda()
                seg_pca_image_flip3 = torch.flip(seg_pca_image,[3])
                seg_pca_image_flip3 = seg_pca_image_flip3.cuda()
                
                seg_pca_image = seg_pca_image.cuda()
                output1 = model(seg_pca_image).cpu().data.numpy() 
                output2 = torch.flip(model(seg_pca_image_flip2),[2]).cpu().data.numpy()
                output3 = torch.flip(model(seg_pca_image_flip3),[3]).cpu().data.numpy()
    
            output += output1 + output2 + output3
            
        # output.shape: 16,10,256,256
        for i in range(output.shape[0]):
            pred = output[i]
            
            pred = np.argmax(pred, axis = 0)
            pred = np.uint8(pred)
            pred[pred==0] = 255
            pred = cv2.resize(pred,(500,500),interpolation=cv2.INTER_NEAREST)
            save_path = os.path.join(output_dir, pca_image_paths[i].split("\\")[-1])
            print(save_path)
            cv2.imwrite(save_path, pred)
        
        
if __name__ == "__main__":
    start_time = time.time()
    pca_model_paths = [
        "../model/pca_unetplusplus_resnest101_fold_0_0.784.pth",
        "../model/pca_unetplusplus_resnest101_fold_1_0.785.pth",
        "../model/pca_unetplusplus_resnest101_fold_2_0.790.pth",
        "../model/pca_unetplusplus_resnest101_fold_3_0.782.pth",
        "../model/pca_unetplusplus_resnest101_fold_4_0.785.pth"
        ]
    seg_pca_model_paths = [
        "../model/seg_pca_unetplusplus_resnest101_fold_0_0.785.pth",
        "../model/seg_pca_unetplusplus_resnest101_fold_1_0.786.pth",
        "../model/seg_pca_unetplusplus_resnest101_fold_2_0.791.pth",
        "../model/seg_pca_unetplusplus_resnest101_fold_3_0.797.pth",
        "../model/seg_pca_unetplusplus_resnest101_fold_4_0.783.pth"
        ]
    output_dir = '../data/test/pred'
    pca_image_paths = glob.glob('../data/test/pca_images/*.tif')
    seg_pca_image_paths = glob.glob('../data/test/seg_pca_images/*.tif')
    batch_size = 1
    num_workers = 8
    test_loader = get_testdataloader(pca_image_paths, seg_pca_image_paths, batch_size, False, 8)
    test(pca_model_paths, seg_pca_model_paths, output_dir, test_loader)
    print((time.time()-start_time)/60**1)