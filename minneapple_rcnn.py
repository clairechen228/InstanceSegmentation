"""
MinneApple - MaskRCNN(tutorial)
Claire Chen
09/28/2021 
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import math
import random
import copy
from torch.utils.tensorboard import SummaryWriter
import transforms as tf

## Parameters
batch_size_train = 5 #total 670
batch_size_test = 9
num_classes = 2 #background and apples
num_epochs = 20
# sep = -10 #number of validation images
path_load = '/apple'
path_save = '/results'


#####################################
# Class that takes the input instance masks
# and extracts bounding boxes on the fly
#####################################
class AppleDataset(object):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        # Load all image and mask files, sorting them to ensure they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))
          
    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)     # Each color of mask corresponds to a different instance with 0 being the background

        # Convert the PIL image to np array
        mask = np.array(mask)
        obj_ids = np.unique(mask)

        # Remove background id
        obj_ids = obj_ids[1:]

        # Split the color-encoded masks into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # Get bbox coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        h, w = mask.shape
        for ii in range(num_objs):
            pos = np.where(masks[ii])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmin == xmax or ymin == ymax:
                continue

            xmin = np.clip(xmin, a_min=0, a_max=w)
            xmax = np.clip(xmax, a_min=0, a_max=w)
            ymin = np.clip(ymin, a_min=0, a_max=h)
            ymax = np.clip(ymax, a_min=0, a_max=h)
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # There is only one class (apples)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = (boxes)
        target["labels"] = (labels)
        target["masks"] = (masks)
        target["image_id"] = (image_id)
        target["area"] = (area)
        target["iscrowd"] = (iscrowd)
            
        T = transforms.Compose([transforms.ToTensor()])
        img = T(img)


        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.imgs[idx]



########################################
#Constructs a Mask R-CNN model with a ResNet-50-FPN backbone.
########################################
def get_model_instance_segmentation(num_classes):
  # load an instance segmentation model pre-trained on COCO
  #a Mask R-CNN model with a ResNet-50-FPN backbone pretraind on COCO
  # pretrained (bool): If True, returns a model pre-trained on COCO train2017
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  # now get the number of input features for the mask classifier
  in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
  hidden_layer = 256
  # and replace the mask predictor with a new one
  model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

  return model
#################################### 



def main():
    # Connect to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    # Load Dataset
    train_data = AppleDataset(os.path.join(path_load,'train'), train = True)
    test_data = AppleDataset(os.path.join(path_load,'validate'), train = False)
    train_dataloader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, collate_fn=utils.collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False, collate_fn=utils.collate_fn)
    
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    writer = SummaryWriter(log_dir=path_save)
    
    # Load weights
    path_weights = '/home/isleri/chen6242/MaskRCNN_ver1/model/1001/after382/9.pth'
    weights = torch.load(path_weights)
    start_epoch = weights['epoch']
    model.load_state_dict(weights['model'])
    optimizer.load_state_dict(weights['optimizer'])
    lr_scheduler.load_state_dict(weights['lr_scheduler'])
    
     
    # Training & Testing
    for epoch in range(start_epoch+1, start_epoch+num_epochs):
    # train for one epoch, printing every 10 iterations
        metric_logger, loss_dict, losses = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        #save paremeters
        filename = str(epoch) + '.pth'
        torch.save({'epoch' : epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'loss_dict': loss_dict
                    }, os.path.join(path_save, filename))
        writer.add_scalar('Sum_losses/train', losses, epoch)
        writer.add_scalar('loss_classifier/train', loss_dict['loss_classifier'], epoch)
        writer.add_scalar('loss_box_reg/train', loss_dict['loss_box_reg'], epoch)
        writer.add_scalar('loss_mask/train', loss_dict['loss_mask'], epoch)
        writer.add_scalar('loss_objectness/train', loss_dict['loss_objectness'], epoch)
        writer.add_scalar('loss_rpn_box_reg/train', loss_dict['loss_rpn_box_reg'], epoch)

        # evaluate on the test dataset
        coco_evaluator = evaluate(model, test_dataloader, device=device)

    writer.close()
    print('DONE')

if __name__ == "__main__":
    main()
