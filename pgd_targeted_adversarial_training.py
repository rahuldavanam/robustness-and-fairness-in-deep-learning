import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, BatchSampler, Sampler

import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import time

from models import *

learning_rate = 0.1
epsilon = 0.0314
k = 7
alpha = 0.00784
batch_size_per_class = 9

file_name = 'pgd_targeted_adversarial_training'

# CUDA agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Macbook device agnostic code
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Custom dataloader function implementation to load equal number of images from all classes
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size_per_class):
        self.dataset = dataset
        self.batch_size_per_class = batch_size_per_class
        self.num_classes = 10  # CIFAR-10 has 10 classes
        self.num_batches = len(self.dataset) // (self.batch_size_per_class * self.num_classes)

        # Create a list of indices for each class
        self.class_indices = [np.where(np.array(self.dataset.targets) == i)[0] for i in range(self.num_classes)]

    def __iter__(self):
        batch = []
        for _ in range(self.num_batches):
            for class_idx in range(self.num_classes):
                indices = np.random.choice(self.class_indices[class_idx], self.batch_size_per_class, replace=False)
                batch.extend(indices)

            np.random.shuffle(batch)
            yield batch
            batch = []

    def __len__(self):
        return self.num_batches

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

balanced_sampler = BalancedBatchSampler(train_dataset, batch_size_per_class)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=balanced_sampler, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

# After defining train_loader
# print(f"Number of batches: {len(balanced_sampler)}")

# Implementation of targeted attack
class LinfPGDTargetAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, target_labels):  # Perturb - make someone unsettled or anxious
        x = x_natural.detach()  # What does it do? It shouldn't propogarte back to x_nat
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, target_labels)
            grad = torch.autograd.grad(loss, [x])[0] 
            # print(grad)
            x = x.detach() - alpha * torch.sign(grad.detach()) 
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

# Custom function to generate target labels for misclassification
def generate_target_labels(labels):
    num_classes = 10
    batch_size_per_class = 9
    target_labels = torch.zeros_like(labels)

    for class_idx in range(num_classes):
        class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
        targets = list(range(num_classes))
        targets.remove(class_idx)

        for i, idx in enumerate(class_indices):
            target_labels[idx] = targets[i % (num_classes - 1)]

    return target_labels

# Function where adversarial examples are created 
def attack(x, y, model, adversary):  
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

adversary = LinfPGDTargetAttack(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        target_labels = generate_target_labels(targets).to(device)
        adv = adversary.perturb(inputs, target_labels)
        adv_outputs = net(adv)
        
        loss = criterion(adv_outputs, targets)
        loss.backward()

        optimizer.step()
        
        train_loss += loss.item() # Converts tensor to python number
        # We're interested in the indices where the max value is present along the 1st dimension (along classes)
        _, predicted = adv_outputs.max(1)  

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 15 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial train loss:', loss.item())

    print(f"Total processed samples: {total}")
    print('\nTotal adversarial train accuarcy:', 100. * correct / total)
    print('Total adversarial train loss:', train_loss)

def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            benign_outputs = net(inputs)
            loss = criterion(benign_outputs, targets)
            benign_loss += loss.item()

            _, predicted = benign_outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign test loss:', loss.item())

            target_labels = generate_target_labels(targets).to(device)
            adv = adversary.perturb(inputs, target_labels)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':        
    start_time = time.time()

    for epoch in range(0,20):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(epoch)

    end_time = time.time() - start_time

    print(f"Time taken for 20 epochs = {end_time/3600} hours")