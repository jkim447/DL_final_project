import math
import os
import datetime
import csv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import tqdm 

import torchvision
from torchvision import models
from torchvision import transforms
from torchvision import datasets

# in order to pre-train final layers
# import the model w/ pretrained --> True
# change the mean & stdev operation in the dataloader for both train/val
# put the model in inference mode, depending for both train/test
# otherwise change it back if training form scratch


# import resnet
m = models.resnet18(pretrained=True)
m.cuda()
print(m)
TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD = [0.229, 0.224, 0.225]

# The Args object will contain all of our parameters
# If you want to run with different arguments, create another Args object
class Args(object):
  def __init__(self, name='mnist', batch_size=64, test_batch_size=1000,
            epochs=30, lr=0.001, optimizer='sgd', momentum=0.5,
            seed=1, log_interval=100, dataset='mnist',
            data_dir='./tiny_imagenet_challenge/tiny-imagenet-200', model='default',
            cuda=True, bce=False):
    self.name = name # name for this training run. Don't use spaces.
    self.batch_size = batch_size
    self.test_batch_size = test_batch_size # Input batch size for testing
    self.epochs = epochs # Number of epochs to train
    self.lr = lr # Learning rate
    self.optimizer = optimizer # sgd/p1sgd/adam/rms_prop
    self.momentum = momentum # SGD Momentum
    self.seed = seed # Random seed
    self.log_interval = log_interval # Batches to wait before logging
                                     # detailed status. 0 = never
    self.dataset = dataset # mnist/fashion_mnist
    self.data_dir = data_dir
    self.model = model # default/P2Q7DoubleChannelsNet/P2Q7HalfChannelsNet/
                  # P2Q8BatchNormNet/P2Q9DropoutNet/P2Q10DropoutBatchnormNet/
                  # P2Q11ExtraConvNet/P2Q12RemoveLayerNet/P2Q13UltimateNet
    self.cuda = cuda and torch.cuda.is_available()
    # for binary cross entropy
    self.bce = bce

def prepare_imagenet(args):
    #dataset_dir = os.path.join(args.data_dir, args.dataset)
    dataset_dir = args.data_dir
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val/images')
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    print('Preparing dataset ...')
    
    # Training data transform
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

    # Validation data Transform 
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
    ])

    # Create train dataloader
    train_data = datasets.ImageFolder(root=train_dir, 
                                      transform=train_transform)
    
    # create test data loader
    val_data = datasets.ImageFolder(root=val_dir, 
                                    transform=val_transform)
    
    print('Preparing data loaders ...')
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                                    shuffle=True, **kwargs)
    
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, 
                                                  shuffle=True, **kwargs)
    
    return train_data_loader, val_data_loader

def train(args, model, optimizer, train_loader, epoch, total_minibatch_count,
        train_losses, train_accs, train_topk_accs):
    # Training for a full epoch
    
    # change it back to model.train() if training from scratch
    #model.train()
    model.eval()
    correct_count, total_loss, total_acc = 0., 0., 0.
    progress_bar = tqdm.tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # zero-out the gradients
        optimizer.zero_grad()

        # Forward prediction step
        output = model(data)
        
        # find the loss
        loss = F.nll_loss(output, target)

        # do backprop
        loss.backward()
        optimizer.step()
        
        # The batch has ended, determine the accuracy of the predicted outputs
        pred = output.data.max(1)[1]  
        
         # target labels and predictions are categorical values from 0 to 9.
        matches = target == pred
        accuracy = matches.float().mean()
        correct_count += matches.sum()
 
        total_loss += loss.data
        total_acc += accuracy.data
        progress_bar.set_description(
            'Epoch: {} loss: {:.4f}, acc: {:.2f}'.format(
                epoch, total_loss / (batch_idx + 1), total_acc / (batch_idx + 1)))
        #progress_bar.refresh()

 
        if args.log_interval != 0 and total_minibatch_count % args.log_interval == 0:

            train_losses.append(loss.data[0].cpu().numpy())
            train_accs.append(accuracy.data[0].cpu().numpy())
            
            # calculate topk accuracy
            batch_size=target.size(0)
            _, pred_topk = output.topk(5,1,True,sorted=True)
            pred_topk = pred_topk.t()
            correct_topk=pred_topk.eq(target.view(1,-1).expand_as(pred_topk))
            correct_topk = correct_topk[:5].view(-1).float().sum(0,keepdim=True)
            correct_topk = correct_topk.mul_(100.0/batch_size)
            
            train_topk_accs.append(correct_topk.data[0].cpu().numpy())
            
            # write to csv file
            #print("logging train csv now")
            with open(os.path.join(os.getcwd(),'train.csv'),'w') as f:
                csvw=csv.writer(f,delimiter=',')
                for loss,acc,topk_accs in zip(train_losses,train_accs,train_topk_accs):
                    csvw.writerow((loss,acc,topk_accs))

        total_minibatch_count += 1

    return total_minibatch_count

def test(args, model, test_loader, epoch, total_minibatch_count,
        val_losses, val_accs, val_topk_accs):
    # Validation Testing
    model.eval()
    test_loss, correct, topk_correct = 0., 0., 0.
    all_outputs = []
    all_labels = []
    progress_bar = tqdm.tqdm(test_loader, desc='Validation')
    with torch.no_grad():
        for data, target in progress_bar:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            
            test_loss += F.nll_loss(output, target, reduction='sum').data  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            
            correct += (target == pred).float().sum()
           
            # Use for error analysis
            all_outputs.append(correct.data[0].cpu().numpy())
            print(correct.data.cpu().numpy().shape)
            all_labels.append(target.data[0].cpu().numpy())
 
            #calculate topk accuracy
            batch_size=target.size(0)
            _, pred_topk = output.topk(5,1,True,sorted=True)
            pred_topk = pred_topk.t()
            correct_topk=pred_topk.eq(target.view(1,-1).expand_as(pred_topk))
            # this is the sum
            correct_topk = correct_topk[:5].view(-1).float().sum(0,keepdim=True)
            # keep a sum of all the correct in topk for this batch
            topk_correct += correct_topk
    
    # perform error analysis
    all_outputs = np.asarray(all_outputs)
    all_labels = np.asarray(all_labels)
    print(all_outputs.shape)
    print(all_labels.shape)
    # number of classes
    correctness = []
    num_corrects = []
    for i in range(200):
        all_outputs = all_outputs == i
        all_labels = all_labels == i
        correct = all_outputs == all_labels

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    topk_acc = topk_correct/len(test_loader.dataset)

    val_losses.append(test_loss.data[0].cpu().numpy())
    val_accs.append(acc.data[0].cpu().numpy())
    val_topk_accs.append(topk_acc.data[0].cpu().numpy())

    # write to csv file
    #print("logging test csv now")
    with open(os.path.join(os.getcwd(),'test.csv'),'w') as f:
        csvw=csv.writer(f,delimiter=',')
        for loss,acc,topk_accs in zip(val_losses,val_accs,val_topk_accs):
            csvw.writerow((loss,acc,topk_accs))

    progress_bar.clear()
    progress_bar.write(
        '\nEpoch: {} validation test results - Average val_loss: {:.4f}, val_acc: {}/{} ({:.2f}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return acc

class TruncatedBinaryResnet(torch.nn.Module):
  
  def __init__(self, orig_resnet):
    super().__init__()
    self.orig_resnet = orig_resnet
    self.final_linear = torch.nn.Linear(512, 200) # TODO
  
  def forward(self, x):
    x = self.orig_resnet.conv1(x)
    x = self.orig_resnet.bn1(x)
    x = self.orig_resnet.relu(x)
    x = self.orig_resnet.maxpool(x)

    x = self.orig_resnet.layer1(x)
    x = self.orig_resnet.layer2(x)
    x = self.orig_resnet.layer3(x)
    x = self.orig_resnet.layer4(x)

    x = self.orig_resnet.avgpool(x)
    x = x.view(x.size(0), -1)

    # x = self.fc(x) # TODO
    x = self.final_linear(x) # TODO
    x = F.log_softmax(x, dim=1)
    return x

# Run the experiment
def run_experiment(args):

    total_minibatch_count = 0

    # choose seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    # load data
    train_loader, val_loader = prepare_imagenet(args)
    epochs_to_run = args.epochs
    
    # initialize model
    model = TruncatedBinaryResnet(m)
    model.cuda()

    # Choose optimizer
    # change it back to model.parameters() if training from scratch
    optimizer = optim.Adam(model.final_linear.parameters())

    val_acc = 0
    train_losses, train_accs, train_topk_accs = [], [], []
    val_losses, val_accs, val_topk_accs = [], [], []

    for epoch in range(1, epochs_to_run + 1):
        # train for 1 epoch
        #total_minibatch_count = train(args, model, optimizer, train_loader,
        #                            epoch, total_minibatch_count,
        #                            train_losses, train_accs, train_topk_accs)
        # validate progress on test dataset
        val_acc = test(args, model, val_loader, epoch, total_minibatch_count,
                       val_losses, val_accs, val_topk_accs)
    
    
    # error analysis
    # save model output, save model
    #fig, axes = plt.subplots(1,4, figsize=(13,4))
    # plot the losses and acc
    #plt.title(args.name)
    #axes[0].plot(train_losses)
    #axes[0].set_title("Loss")
    #axes[1].plot(train_accs)
    #axes[1].set_title("Acc")
    #axes[2].plot(val_losses)
    #axes[2].set_title("Val loss")
    #axes[3].plot(val_accs)
    #axes[3].set_title("Val Acc")
    
    # Write to csv file
    #with open(os.path.join(run_path + 'train.csv'), 'w') as f:
    #    csvw = csv.writer(f, delimiter=',')
    #    for loss, acc in zip(train_losses, train_accs):
    #        csvw.writerow((loss, acc))

    # Predict and Test
    #images, labels = next(iter(test_loader))
    #if args.cuda:
    #    images, labels = images.cuda(), labels.cuda()
    #output = model(images)
    #predicted = torch.max(output, 1)[1]
    #fig, axes = plt.subplots(1,6)
    #for i, (axis, img, lbl) in enumerate(zip(axes, images, predicted)):
    #    if i > 5:
    #        break
    #    img = img.permute(1,2,0).squeeze()
    #    axis.imshow(img)
    #    axis.set_title(lbl.data)
    #    axis.set_yticklabels([])
    #    axis.set_xticklabels([])
            
    #if args.dataset == 'fashion_mnist' and val_acc > 0.92 and val_acc <= 1.0:
    #    print("Congratulations, you beat the Question 13 minimum of 92"
    #        "with ({:.2f}%) validation accuracy!".format(val_acc))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, kernel_size=8,stride=6)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(750, 400)
        self.fc2 = nn.Linear(400, 200)

    def forward(self, x):
        # F is just a functional wrapper for modules from the nn package
        # see http://pytorch.org/docs/_modules/torch/nn/functional.html
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 750)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

run_experiment(Args())
