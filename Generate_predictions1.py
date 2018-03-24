#Code for generating predictions for evaluation set stage I
from __future__ import print_function, division
import itertools
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import time
import os
import torchsample
import resnet3

data_transforms = {
    'train': transforms.Compose([transforms.Scale((512,256)), 
        transforms.ToTensor(),
        torchsample.transforms.RandomTranslate((0.05,0.1)),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

 ]),
    'eval': transforms.Compose([

        transforms.Scale((512,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
}

def save_checkpoint(state, is_best, filename='predict.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

data_dir = 'GD1'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'eval']}
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=False, num_workers=4)
              for x in ['train', 'eval']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'eval']}
class_names = image_datasets['train'].classes


use_gpu = torch.cuda.is_available()


def predict_model(model, criterion, optimizer, scheduler, num_epochs=1):

    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['eval']:
            #model.eval()
            model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                #_, preds = torch.max(outputs.data, 1)
                #loss = criterion(outputs, labels)
                out=(outputs.data).cpu().numpy()
                #print(out)
                #print(out.shape)
                h_x = F.softmax(outputs).data.squeeze()  
                #print(h_x)
                h_x=(h_x).cpu().numpy()
             
                with open('finfeaturesGDIeval.csv','ab') as f:
                ##with open('finrand.csv','ab') as f:

                    np.savetxt(f,h_x,delimiter=',')
                num, preds = torch.max(outputs.data, 1)
                print(preds)
                pred=(preds).cpu().numpy()

                loss = criterion(outputs, labels)

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model





from torchsample.callbacks import ReduceLROnPlateau

model_ft = resnet3.resnet182(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)


ct = 0
for child in model_ft.children():
    ct += 1
    if ct < 0:
            for param in child.parameters():
                param.requires_grad = False


if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

parameters = itertools.ifilter(lambda p: p.requires_grad,model_ft.parameters())


optimizer_ft = optim.Adam(parameters,lr=0.0001)

resume='GD1.pth.tar'

if resume:
    if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            print('best_prec')
            print(best_prec1)
            model_ft.load_state_dict(checkpoint['state_dict'])

            optimizer_ft.load_state_dict(checkpoint['optimizer'])
            #model_ft.eval()
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    else:
            print("=> no checkpoint found at '{}'".format(resume))


exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)




model_ft = predict_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)


#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=1)



