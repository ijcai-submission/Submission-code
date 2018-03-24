import cv2,torch
import io

from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

import resnet3
import torch.nn.functional as F

import torch.nn as nn
import os



net=resnet3.resnet182(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
net.avgpool = nn.AdaptiveAvgPool2d(1)
net.eval()
resume='GD1.pth.tar'
if resume:
  if os.path.isfile(resume):
          print("=> loading checkpoint '{}'".format(resume))
          checkpoint = torch.load(resume)
          start_epoch = checkpoint['epoch']
          best_prec1 = checkpoint['best_prec1']
          net.load_state_dict(checkpoint['state_dict'])
          print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
          finalconv_name = 'layer4'

  else:
          print("=> no checkpoint found at '{}'".format(resume))



# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

#initially tfo scale 512, 256
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale((512,256)),
   transforms.ToTensor(),
   normalize
])
import os
directory='GD1/train/1'
directory2='GD2/train/1'



for filename in os.listdir(directory):


  if filename.endswith(".png") or filename.endswith(".jpg"): 
      path=os.path.join(directory, filename)
      path2=os.path.join(directory2,filename)

      img_pil = Image.open(path).convert('RGB')

      img_pil.save('testtrainspo.jpg')

      img_tensor = preprocess(img_pil)
      img_variable = Variable(img_tensor.unsqueeze(0))
      logit = net(img_variable)

# download the imagenet category list
      #classes = {int(key):value for (key, value)
      #in requests.get(LABELS_URL).json().items()}
      h_x = F.softmax(logit).data.squeeze()
      probs, idx = h_x.sort(0, True)

# output the prediction
      #for i in range(0, 2):
        #print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
      #print('traingen')

# generate class activation mapping for the top1 prediction
      CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
      #print('CAM')
      #print(CAMs)


      #print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])

      img = cv2.imread('testtrainspo.jpg')
      height, width, _ = img.shape
      heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
      heatmap=cv2.resize(CAMs[0],(width, height))
      heat=np.zeros((1025,290,3))
      #heat=np.zeros((513,256,3))
      heat[:,:,0]=heatmap
      heat[:,:,1]=heatmap
      heat[:,:,2]=heatmap



      result = np.multiply(heat,img)
      result=result/255
        
      img_path = path2.replace(".png", "_refined.png")

      cv2.imwrite(img_path, result)
      #print(img_path)
      continue
  else:
      continue

