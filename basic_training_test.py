import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *
from advertorch.attacks import LinfPGDAttack

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

file_name = 'basic_training'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
checkpoint = torch.load('./checkpoint/' + file_name)
net.load_state_dict(checkpoint['net'])

adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=0.0314, nb_iter=7, eps_iter=0.00784, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
criterion = nn.CrossEntropyLoss()

y_pred = []
y_pred_adv = []

def test():
    print('\n[ Test Start ]')
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        y_pred.append(predicted.cpu())  # Move to CPU to avoid device mismatch later
        benign_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign test loss:', loss.item())

        adv = adversary.perturb(inputs, targets)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets)
        adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        y_pred_adv.append(predicted.cpu())  # Move to CPU to avoid device mismatch later
        adv_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuracy:', 100. * benign_correct / total)
    print('Total adversarial test accuracy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

test()

y_pred_tensor = torch.cat(y_pred).to(device)
y_pred_adv_tensor = torch.cat(y_pred_adv).to(device)

class_names = test_dataset.classes
targets_tensor = torch.tensor(test_dataset.targets).to(device)

# Plotting a benign confusion matrix
confmat = ConfusionMatrix(task='multiclass', num_classes=len(class_names)).to(device)
confmat_tensor = confmat(preds=y_pred_tensor, target=targets_tensor)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.cpu().numpy(), 
    class_names=class_names,
    figsize=(10, 7)
)

# Plotting an adversarial confusion matrix
confmat_adv = ConfusionMatrix(task='multiclass', num_classes=len(class_names)).to(device)
confmat_adv_tensor = confmat_adv(preds=y_pred_adv_tensor, target=targets_tensor)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_adv_tensor.cpu().numpy(), 
    class_names=class_names,
    figsize=(10, 7)
)