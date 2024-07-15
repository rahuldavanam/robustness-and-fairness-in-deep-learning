import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *
from advertorch.attacks import LinfPGDAttack

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import random

learning_rate = 0.1
epsilon = 0.0314
k = 7
alpha = 0.00784
batch_size_per_class = 9

file_name = 'loss_norms_untargeted_square_root'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

# Implementation of targeted attack
class LinfPGDTargetAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, target_labels):  # Perturb - make someone unsettled or anxious
        x = x_natural.detach() 
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
    target_labels = labels.clone().detach()
    
    for i in range(len(labels)):
        targets = list(range(num_classes))
        targets.remove(labels[i].item())
        target_labels[i] = random.choice(targets)
        
    return target_labels

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
checkpoint = torch.load('./checkpoint/' + file_name)
net.load_state_dict(checkpoint['net'])

target_adversary = LinfPGDTargetAttack(net)
untarget_adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=0.0314, nb_iter=7, eps_iter=0.00784, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
criterion = nn.CrossEntropyLoss()

y_pred = []
y_pred_target_adv = []
y_pred_untarget_adv = []

def test():
    print('\n[ Test Start ]')
    net.eval()
    benign_loss = 0
    target_adv_loss = 0
    untarget_adv_loss = 0
    benign_correct = 0
    target_adv_correct = 0
    untarget_adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        # Accuracy and loss on BENIGN data
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        y_pred.append(predicted.cpu())  # Move to CPU to avoid device mismatch later
        benign_correct += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign test loss:', loss.item())

        # Accuracy and loss on TARGETED adversarial data
        target_labels = generate_target_labels(targets).to(device)
        adv = target_adversary.perturb(inputs, target_labels)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets)
        target_adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        y_pred_target_adv.append(predicted.cpu())  # Move to CPU to avoid device mismatch later
        target_adv_correct += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print('Current adversarial test accuracy on targeted attacks:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial test loss on targeted attacks:', loss.item())

        # Accuracy and loss on UNTARGETED adversarial data
        adv = untarget_adversary.perturb(inputs, targets)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets)
        untarget_adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        y_pred_untarget_adv.append(predicted.cpu())  # Move to CPU to avoid device mismatch later
        untarget_adv_correct += predicted.eq(targets).sum().item()

        if batch_idx % 20 == 0:
            print('Current adversarial test accuracy on untargeted attacks:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial test loss on untargeted attacks:', loss.item())

    print('\nTotal benign test accuracy:', 100. * benign_correct / total)
    print('Total adversarial test accuracy on targeted attacks:', 100. * target_adv_correct / total)
    print('Total adversarial test accuracy on untargeted attacks:', 100. * untarget_adv_correct / total)
    print('\nTotal benign test loss:', benign_loss)
    print('Total adversarial test loss on targeted attacks:', target_adv_loss)
    print('Total adversarial test loss on untargeted attacks:', untarget_adv_loss)

def plot_confusion_matrix(true_labels, predicted_labels, classes, title, filename):
    cm = confusion_matrix(true_labels, predicted_labels)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    plt.figure(figsize=(15, 10))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    test()

    y_pred_tensor = torch.cat(y_pred)
    y_pred_target_adv_tensor = torch.cat(y_pred_target_adv)
    y_pred_untarget_adv_tensor = torch.cat(y_pred_untarget_adv)

    class_names = test_dataset.classes
    targets_tensor = torch.tensor(test_dataset.targets)

    # Benign confusion matrix
    plot_confusion_matrix(targets_tensor, y_pred_tensor, class_names, title="Confusion matrix of l0.5 loss untargeted training on benign data", filename="Confusion_matrix_of_l0.5_loss_untargeted_training_on_benign_data.png")
    
    # Untargeted confusion matrix
    plot_confusion_matrix(targets_tensor, y_pred_untarget_adv_tensor, class_names, title="Confusion matrix of l0.5 loss untargeted training on untargeted data", filename="Confusion_matrix_of_l0.5_loss_untargeted_training_on_untargeted_data.png")
   
    # Targeted confusion matrix
    plot_confusion_matrix(targets_tensor, y_pred_target_adv_tensor, class_names, title="Confusion matrix of l0.5 loss untargeted training on targeted data", filename="Confusion_matrix_of_l0.5_loss_untargeted_training_on_targeted_data.png")