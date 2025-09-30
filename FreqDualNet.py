from __future__ import annotations
import torch
#import torchvision as tv
import torch.nn as nn
from torch.nn import functional as F
import time
from PIL import Image
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from torchsummary import summary as model_summary
import glob
import nibabel as nib
import skimage
import sklearn
from sklearn import model_selection
import monai
# GPU 할당해주는 코드 교수님께서 알려주신거
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# GPU 할당해주는 코드 교수님께서 알려주신거
import datetime as dt
save_dir = "./FreqDualNet"
os.makedirs(save_dir, exist_ok=True)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    Resized,
    NormalizeIntensityd,
    ToTensord,
    AsDiscrete,
    RandRotated,
    RandRotate,
)
from monai.config import print_config
from typing import Optional
from monai.metrics import MAEMetric

from tqdm import tqdm


print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}')
print(f'torch.version.cuda: {torch.version.cuda}')
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
print()
monai.config.print_config()

### HYPER PARAMETER ###

RANDOM_SEED = 55
#IMAGE_SIZE = (256, 256, 256)
IMAGE_SIZE = (224,224,224)  
BATCH_SIZE = 1
NUM_CLASS = 3
NUM_CLASS_ONE_HOT = 2
EPOCHS = 600
test_ratio, val_ratio = 0.1, 0.20

MODEL_SAVE = True
if MODEL_SAVE:
    model_dir1 = '/home/gail11/orgunetr/save_model'
    model_dir2 = 'fft3_results'
    MODEL_SAVE_PATH = os.path.join(model_dir1, model_dir2)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'


USE_MY_DATA = True

if not USE_MY_DATA:
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(f"root dir is: {root_dir}")
    
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

else:  
    #For Downloading Kidney Tumor Dataset, please visit "https://kits19.grand-challenge.org/"
    data_dir = r"/home/gail11/orgunetr/Dataset/KITS19_224"  
train_images = sorted(
    glob.glob(os.path.join(data_dir, "normalized", "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(data_dir, "segmentation", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
train_data_dicts, val_data_dicts = data_dicts[:-9], data_dicts[-9:]
train_data_dicts[0]





TrainSet, TestSet = model_selection.train_test_split(data_dicts, test_size=test_ratio, random_state=RANDOM_SEED)
TrainSet, ValSet = model_selection.train_test_split(TrainSet, test_size=val_ratio, random_state=RANDOM_SEED)
print('TrainSet:', len(TrainSet), 'ValSet:', len(ValSet), 'TestSet:', len(TestSet))

for i in range(3):
    sample_img = nib.load(TrainSet[i]['image']).get_fdata()
    sample_mask = nib.load(TrainSet[i]['label']).get_fdata()
    print(f"[sample {i+1}] {os.path.basename(TrainSet[i]['image'])} {os.path.basename(TrainSet[i]['label'])}")
    print(sample_img.shape, sample_img.dtype, np.min(sample_img), np.max(sample_img))
    print(sample_mask.shape, sample_mask.dtype, np.unique(sample_mask))


from monai.transforms.compose import Transform, MapTransform

class MinMax(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] -= np.min(d[key])
            d[key] /= np.max(d[key])
        return d

loader = LoadImaged(keys=("image",'label'), image_only=False)
ensure_channel_first = EnsureChannelFirstd(keys=["image",'label'])
orientation = Orientationd(keys=["image",'label'], axcodes="RAS")
resize_img = Resized(keys=["image",], spatial_size=(IMAGE_SIZE), mode='trilinear')
resize_mask = Resized(keys=['label',], spatial_size=(IMAGE_SIZE), mode='nearest-exact')
# normalize = NormalizeIntensityd(keys=["image",])
minmax = MinMax(keys=['image',])


from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Resized, NormalizeIntensityd, ToTensord, RandRotated
from monai.transforms import RandFlipd, RandAffined, RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd


transforms = Compose([
    LoadImaged(keys=("image",'label'), image_only=False),
    EnsureChannelFirstd(keys=["image",'label']),
    Orientationd(keys=["image",'label'], axcodes="RAS"),    
    Resized(keys=["image"], spatial_size=IMAGE_SIZE, mode='trilinear'),
    Resized(keys=["label"], spatial_size=IMAGE_SIZE, mode='nearest'),
    MinMax(keys=['image',]),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    # RandAffined(keys=["image", "label"], prob=0.3, rotate_range=(0.1,0.1,0.1), scale_range=(0.1,0.1,0.1)),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.1),
    RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),
    ToTensord(keys=["image", "label"]),
])

transforms_val = Compose([
    LoadImaged(keys=("image",'label'), image_only=False),
    EnsureChannelFirstd(keys=["image",'label']),
    Orientationd(keys=["image",'label'], axcodes="RAS"),    
    Resized(keys=["image"], spatial_size=IMAGE_SIZE, mode='trilinear'),
    Resized(keys=["label"], spatial_size=IMAGE_SIZE, mode='nearest'),
    MinMax(keys=['image',]),
    ToTensord(keys=["image", "label"]),    
])


SampleSet = transforms(TestSet[:3])


for i in range(3):
    sample_img = nib.load(TrainSet[i]['image']).get_fdata()
    sample_mask = nib.load(TrainSet[i]['label']).get_fdata()
    print(f"[sample {i+1}] {os.path.basename(TrainSet[i]['image'])} {os.path.basename(TrainSet[i]['label'])}")
    print(sample_img.shape, sample_img.dtype, np.min(sample_img), np.max(sample_img))
    print(sample_mask.shape, sample_mask.dtype, np.unique(sample_mask))

from monai.transforms.compose import Transform, MapTransform

class MinMax(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] -= np.min(d[key])
            d[key] /= np.max(d[key])
        return d

loader = LoadImaged(keys=("image",'label'), image_only=False)
ensure_channel_first = EnsureChannelFirstd(keys=["image",'label'])
orientation = Orientationd(keys=["image",'label'], axcodes="RAS")
resize_img = Resized(keys=["image",], spatial_size=(IMAGE_SIZE), mode='trilinear')
resize_mask = Resized(keys=['label',], spatial_size=(IMAGE_SIZE), mode='nearest-exact')
# normalize = NormalizeIntensityd(keys=["image",])
minmax = MinMax(keys=['image',])




SampleSet = transforms(TestSet[:3])

for i in range(3):
    sample_img = SampleSet[i]['image']
    sample_mask = SampleSet[i]['label']
    print(f"[sample {i+1}]")
    print(sample_img.shape, sample_img.dtype, torch.min(sample_img), torch.max(sample_img))
    print(sample_mask.shape, sample_mask.dtype, torch.unique(sample_mask))



ncols, nrows = 10, 6
interval = int(IMAGE_SIZE[-1]//(ncols*nrows/2))
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols,nrows))
cnt1, cnt2 = 0, 0
for i in range(nrows):
    for j in range(ncols):
        if i%2 == 0:
            axes[i,j].imshow(SampleSet[0]['image'][0,:,:,cnt1], cmap='gray')
            cnt1+=interval
        else:
            axes[i,j].imshow(SampleSet[0]['label'][0,:,:,cnt2], cmap='gray')
            cnt2+=interval
        axes[i,j].axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.show()  



from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

train_ds = CacheDataset(
    data=TrainSet,
    transform=transforms,
    cache_num=2,
    cache_rate=0.5,
    num_workers=0)
val_ds = CacheDataset(
    data=ValSet, transform=transforms_val, cache_num=2, cache_rate=0.5, num_workers=0)
test_ds = CacheDataset(
    data=TestSet, transform=transforms_val, cache_num=2, cache_rate=0.5, num_workers=0)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(
    test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch import Tensor





print('done')



import sys
import os
from Freq_Net import *



from monai.losses import DiceCELoss, DiceLoss
from monai.losses.dice import one_hot



def BinaryOutput(output, keepdim=True):
    shape = output.shape
    argmax_idx = torch.argmax(output, axis=1, keepdim=True)
    argmax_oh = F.one_hot(argmax_idx, num_classes=NUM_CLASS_ONE_HOT)
    if keepdim:
        argmax_oh = torch.squeeze(argmax_oh, dim=1)
    if len(shape) == 5:
        argmax_oh = argmax_oh.permute(0,4,1,2,3)
    elif len(shape) == 4:
        argmax_oh = argmax_oh.permute(0,3,1,2)
    
    return argmax_oh

print("done")



from monai.losses import DiceCELoss, DiceLoss
from monai.losses.dice import one_hot


model = SwinUNETR(
    img_size=(224,224,224),
    in_channels=1,
    out_channels=4,
    feature_size=12,
    num_heads=(1, 2, 1, 2),
    depths=(3,6,12,24),
    use_checkpoint=True,).to(DEVICE)

#_model = model.to(DEVICE)
#model = nn.DataParallel(_model).to(DEVICE)
# model_summary(model, (1,*IMAGE_SIZE), device=DEVICE.type)



torch.backends.cudnn.benchmark = True # ??

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5*5)
LossFunction = monai.losses.DiceCELoss(include_background=False, to_onehot_y=False, softmax=True)
# LossFunction = monai.losses.DiceLoss(include_background=False, to_onehot_y=False, softmax=True)
#MAELoss = nn.L1Loss()
# MetricDice = monai.metrics.DiceMetric(include_background=False, reduction="mean")
from monai.metrics import DiceMetric
# from monai.utils import one_hot

MetricDice = DiceMetric(include_background=False, reduction="none")

def compute_valid_dice_with_monai(pred, target):
    """
    pred, target: (B, C, H, W[, D])
    pred: one-hot or binary prediction
    target: one-hot GT
    Returns: float, mean dice score over valid classes (GT > 0)
    """
    MetricDice.reset()
    MetricDice(pred, target)
    dice_per_channel = MetricDice.aggregate().detach()  # shape: (B, C)

    dims = tuple(range(2, target.ndim))  # (2,3) or (2,3,4)
    target_sum = target.sum(dim=dims)
    valid_mask = (target_sum > 0).float()  # shape: (B, C)

    masked_dice = dice_per_channel * valid_mask
    num_valid = valid_mask.sum(dim=1).clamp(min=1)  

    mean_dice = (masked_dice.sum(dim=1) / num_valid).mean().item()
    return mean_dice


def BinaryOutput(output, keepdim=True):
    shape = output.shape
    argmax_idx = torch.argmax(output, axis=1, keepdim=True)
    argmax_oh = F.one_hot(argmax_idx, num_classes=NUM_CLASS_ONE_HOT)
    if keepdim:
        argmax_oh = torch.squeeze(argmax_oh, dim=1)
    if len(shape) == 5:
        argmax_oh = argmax_oh.permute(0,4,1,2,3)
    elif len(shape) == 4:
        argmax_oh = argmax_oh.permute(0,3,1,2)
    
    return argmax_oh

print("done")


from torch.amp import GradScaler
from torch.amp import autocast
scaler = GradScaler('cuda')

ACCUMULATION_STEPS = 4  

def train(epoch, train_loader):
    mean_epoch_loss = 0
    mean_dice_score_organ = 0
    mean_dice_score_tumor = 0
    #print(dt.datetime.now(), "   epoch_iterator 전")
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X EPOCHS) (loss=X.X) (dice score=%.5f)", 
        dynamic_ncols=True)

    data_check_iter = 0
    for step, batch in enumerate(epoch_iterator):

        x, y = (batch["image"].to(DEVICE), batch["label"])
        #print("y.shape : ", y.shape)
        y_organ = torch.zeros(y.shape)
        y_organ[y==1] = 1
        y_organ[y==2] = 1
        print("y_organ.shape : ", y_organ.shape)
        y_organ = one_hot(y_organ, num_classes=NUM_CLASS_ONE_HOT)
        # print("y_organ.shape : ", y_organ[:,:,:,])

        y_organ = y_organ.to(DEVICE)

        y_tumor = torch.zeros(y.shape)
        y_tumor[y==1] = 0
        y_tumor[y==2] = 1
        y_tumor = one_hot(y_tumor, num_classes=NUM_CLASS_ONE_HOT)
        y_tumor = y_tumor.to(DEVICE)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            pred = model(x)
            pred_organ = pred[:,0:2,:,:,:]
            pred_tumor = pred[:,2:4,:,:,:]
            pred_organ = pred_organ.to(DEVICE)
            pred_tumor = pred_tumor.to(DEVICE)
            loss_organ = LossFunction(pred_organ, y_organ)
            loss_tumor = LossFunction(pred_tumor, y_tumor)
            loss = loss_organ * 0.68 + loss_tumor * 0.32

        loss = loss / ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(epoch_iterator):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        mean_epoch_loss += loss.item() * ACCUMULATION_STEPS

        bi_organ = BinaryOutput(pred_organ)
        dice_score_organ = compute_valid_dice_with_monai(bi_organ, y_organ)
       
        mean_dice_score_organ += dice_score_organ

        MetricDice.reset()

        bi_tumor = BinaryOutput(pred_tumor)
        dice_score_tumor = compute_valid_dice_with_monai(bi_tumor, y_tumor)
        
        mean_dice_score_tumor += dice_score_tumor

        MetricDice.reset()
        
        epoch_iterator.set_description(
            "Training (%d / %d EPOCHS) (loss= %2.4f) (dice score organ=%.4f) (dice score tumor = %.4f)" 
            % (epoch, EPOCHS, loss.item(), dice_score_organ, dice_score_tumor))
        
    
    mean_epoch_loss /= len(epoch_iterator)    
    mean_dice_score_organ /= len(epoch_iterator)
    mean_dice_score_tumor /= len(epoch_iterator)

    print("mean epoch loss : ", mean_epoch_loss)
    print("mean dice score organ : ", mean_dice_score_organ)
    print("mean dice score tumor : ", mean_dice_score_tumor)

    return mean_epoch_loss, mean_dice_score_organ, mean_dice_score_tumor





def evaluate(epoch, test_loader):
    model.eval() 
    mean_epoch_loss = 0
    mean_dice_score_organ = 0
    mean_dice_score_tumor = 0
    epoch_iterator = tqdm(
        test_loader, desc="Evaluating (X / X EPOCHS) (loss=X.X) (dice score=%.5f)", 
        dynamic_ncols=True)
    
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            x, y = (batch["image"].to(DEVICE), batch["label"])

            y_organ = torch.zeros(y.shape)
            y_organ[y==1] = 1
            y_organ[y==2] = 1
            y_organ = one_hot(y_organ, num_classes=NUM_CLASS_ONE_HOT)
            y_organ = y_organ.to(DEVICE)

            y_tumor = torch.zeros(y.shape)
            y_tumor[y==1] = 0
            y_tumor[y==2] = 1
            y_tumor = one_hot(y_tumor, num_classes=NUM_CLASS_ONE_HOT)
            y_tumor = y_tumor.to(DEVICE)

            with autocast(device_type='cuda'):
                pred = model(x)
                pred_organ = pred[:,0:2,:,:,:]        
                pred_tumor = pred[:,2:4,:,:,:]
                pred_organ = pred_organ.to(DEVICE)
                pred_tumor = pred_tumor.to(DEVICE)
                
                loss_organ = LossFunction(pred_organ, y_organ)
                loss_tumor = LossFunction(pred_tumor, y_tumor)
            
            loss = loss_organ * 0.68 + loss_tumor * 0.32
            
            mean_epoch_loss += loss.item()

            bi_organ = BinaryOutput(pred_organ)
            dice_score_organ = compute_valid_dice_with_monai(bi_organ, y_organ)
            # MetricDice(bi_organ, y_organ)
            # dice_score_organ = MetricDice.aggregate().item()
            mean_dice_score_organ += dice_score_organ

            MetricDice.reset()

            bi_tumor = BinaryOutput(pred_tumor)
            dice_score_tumor = compute_valid_dice_with_monai(bi_tumor, y_tumor)
            # MetricDice(bi_tumor, y_tumor)
            # dice_score_tumor = MetricDice.aggregate().item()
            mean_dice_score_tumor += dice_score_tumor         

            MetricDice.reset()   
            
            epoch_iterator.set_description(
                "Evaluating (%d / %d EPOCHS) (loss= %2.4f) (dice organ=%.5f) (dice tumor = %.5f)" 
                % (epoch, EPOCHS, loss.item(), dice_score_organ, dice_score_tumor))

        mean_epoch_loss /= len(epoch_iterator)
        mean_dice_score_organ /= len(epoch_iterator)
        mean_dice_score_tumor /= len(epoch_iterator)

        print("mean epoch loss : ", mean_epoch_loss)
        print("mean dice score organ : ", mean_dice_score_organ)
        print("mean dice score tumor : ", mean_dice_score_tumor)
        
       #MetricDice.reset() # reset the status for next validation round
        
    return mean_epoch_loss, mean_dice_score_organ, mean_dice_score_tumor   


losses = {'train':[], 'val':[]}
dice_scores_organ = {'train':[], 'val':[]}
dice_scores_tumor = {'train':[], 'val':[]}
best_metric, best_epoch = -1, -1
import os


iter = 0

for epoch in range(1, EPOCHS+1):
    train_loss, train_dice_score_organ, train_dice_score_tumor = train(epoch, train_loader)
    val_loss, val_dice_score_organ, val_dice_score_tumor = evaluate(epoch, val_loader)
    losses['train'].append(train_loss)
    losses['val'].append(val_loss)
    dice_scores_organ['train'].append(train_dice_score_organ)
    dice_scores_organ['val'].append(val_dice_score_organ)
    dice_scores_tumor['train'].append(train_dice_score_tumor)
    dice_scores_tumor['val'].append(val_dice_score_tumor)

    if dice_scores_tumor['val'][-1] > best_metric:
        if epoch > 100:
            best_metric = dice_scores_tumor['val'][-1]
            best_epoch = epoch
            print(f'Best record! [{epoch}] Test Loss: {val_loss:.6f}, Dice organ : {val_dice_score_organ:.6f}, Dice tumor : {val_dice_score_tumor:.6f}')
            if MODEL_SAVE:
                model_name = f'./{best_epoch}_{best_metric}.pth'
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_name))
                print('saved model')
            
with open("res_losses_train_fft_3_test_0.1.txt", "w") as output:
    for item in losses['train']:
        output.write("%f\n" % item)

with open("res_losses_val_fft_3_test_0.1.txt", "w") as output:
    for item in losses['val']:
        output.write("%f\n" % item)
    
with open("res_dice_score_organ_train_fft_3_test_0.1.txt", "w") as output:
    for item in dice_scores_organ['train']:
        output.write("%f\n" % item)
    
with open("res_dice_score_organ_val_fft_3_test_0.1.txt", "w") as output:
    for item in dice_scores_organ['val']:
        output.write("%f\n" % item)
    
with open("res_dice_score_tumor_train_fft_3_test_0.1.txt", "w") as output:
    for item in dice_scores_tumor['train']:
        output.write("%f\n" % item)

with open("res_dice_score_tumor_val_fft_3_test_0.1.txt", "w") as output:
    for item in dice_scores_tumor['val']:
        output.write("%f\n" % item)
    

print("done")


epochs = [i for i in range(len(losses['train']))]
train_loss = losses['train']
val_loss = losses['val']
train_dice_organ = dice_scores_organ['train']
val_dice_organ = dice_scores_organ['val']
train_dice_tumor = dice_scores_tumor['train']
val_dice_tumor = dice_scores_tumor['val']

fig , ax = plt.subplots(1,3)
fig.set_size_inches(18,6)

ax[0].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[0].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[0].set_title('Training & Validation Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(epochs , train_dice_organ , 'go-' , label = 'Training Dice score organ')
ax[1].plot(epochs , val_dice_organ , 'ro-' , label = 'Validation Dice score organ')
ax[1].set_title('Training & Validation Dice score for organ')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Dice score organ")

ax[2].plot(epochs , train_dice_tumor , 'go-' , label = 'Training Dice score tumor')
ax[2].plot(epochs , val_dice_tumor , 'ro-' , label = 'Validation Dice score tumor')
ax[2].set_title('Training & Validation Dice score for tumor')
ax[2].legend()
ax[2].set_xlabel("Epochs")
ax[2].set_ylabel("Dice score tumor")

plt.show()
plt.savefig('results_with_small_size_224_lr1e-3_kits_fft_3_test_0.1.png')



pred_dict = {'input':[], 'target':[], 'output_organ':[], 'output_tumor':[]}

if MODEL_SAVE:
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, model_name)))

model.to('cpu')
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        img, target = data["image"].cpu(), data["label"].cpu()

        output = model(img).detach().cpu()
        output_organ = output[:,0:2,:,:,:]
        output_tumor = output[:,2:4,:,:,:]
        output_organ = torch.argmax(output_organ, dim=1)
        output_tumor = torch.argmax(output_tumor, dim=1)
        
        pred_dict['input'].append(img)
        pred_dict['target'].append(target)
        pred_dict['output_organ'].append(output_organ)
        pred_dict['output_tumor'].append(output_tumor)
        os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)
        numpy_target = target.numpy()
        filename = 'res_target_patch_4_lr1e-3_fft_3_test_0.1' + str(i) + ".npy"
        np.save(os.path.join(save_dir, "predictions", filename), numpy_target)
        numpy_img = img.numpy()
        filename = 'res_image_patch_4_lr1e-3_fft_3_test_0.1' + str(i) + ".npy"
        np.save(os.path.join(save_dir, "predictions", filename), numpy_img)

        numpy_organ = output_organ.numpy()
        filename = 'res_organ_prediction_patch_4_lr1e-3_fft_3_test_0.1' + str(i) + ".npy"
        np.save(os.path.join(save_dir, "predictions", filename), numpy_organ)
        numpy_tumor = output_tumor.numpy()
        filename = 'res_tumor_prediction_patch_4_lr1e-3_fft_3_test_0.1' + str(i) + ".npy"
        np.save(os.path.join(save_dir, "predictions", filename), numpy_tumor)
        
        #if i > 10:
            #break


ncols, nrows = 10, 3*3
interval = int(IMAGE_SIZE[-1]//(ncols*nrows/3))
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols,nrows))
cnt1, cnt2, cnt3 = 0, 0, 0
for i in range(nrows):
    for j in range(ncols):
        if i%3 == 0:
            axes[i,j].imshow(pred_dict['input'][0][0,0,:,:,cnt1], cmap='gray')
            cnt1+=interval
        elif i%3 == 1:
            axes[i,j].imshow(pred_dict['target'][0][0,0,:,:,cnt2], cmap='gray')
            cnt2+=interval
        else:
            axes[i,j].imshow(pred_dict['output_organ'][0][0,:,:,cnt3], cmap='gray')
            cnt3+=interval
        axes[i,j].axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.show()  
plt.savefig('res_predictions_organ_transformer_X_lr1e-3_kits_fft_3_test_0.1.png')



ncols, nrows = 10, 3*3
interval = int(IMAGE_SIZE[-1]//(ncols*nrows/3))
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols,nrows))
cnt1, cnt2, cnt3 = 0, 0, 0
for i in range(nrows):
    for j in range(ncols):
        if i%3 == 0:
            axes[i,j].imshow(pred_dict['input'][0][0,0,:,:,cnt1], cmap='gray')
            cnt1+=interval
        elif i%3 == 1:
            axes[i,j].imshow(pred_dict['target'][0][0,0,:,:,cnt2], cmap='gray')
            cnt2+=interval
        else:
            axes[i,j].imshow(pred_dict['output_tumor'][0][0,:,:,cnt3], cmap='gray')
            cnt3+=interval
        axes[i,j].axis('off')
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.show()  
plt.savefig('res_predictions_tumor_transformer_X_lr1e-3_kits_fft_3_test_0.1.png')



print(f"Code Finished")
