# %%
import pickle

import torchvision.models
from torch.utils.data import SubsetRandomSampler
from torchvision.models.vgg import model_urls
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import math
# %%
# Constants
BATCH_SIZE = 32
VAL_SPLIT = 1/5
# %%
# Load model
for k in model_urls.keys():
    model_urls[k] = model_urls[k].replace('https://', 'http://')

model = torchvision.models.vgg11(pretrained=True)
model.eval()
# %%
# Load dataset
# Color images 224×224 pixels of 10 categories.
from torchvision import datasets, transforms
train_data = datasets.ImageFolder('butterflies/train', transforms.ToTensor())
n_data = len(train_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# %%
# Perform standartization of the data: on the training set compute mean
# and standard deviation per color channel over all pixels and all images in the training set.
# Think how to do it incrementally with mini-batches, not loading the whole dataset into memory at once.
mean = 0.0
for images, _ in train_loader:
    images = images.view(images.size(0), images.size(1), -1)
    mean += images.mean(2).sum(0)

mean = mean / n_data

var = 0.0
for images, _ in train_loader:
    images = images.view(images.size(0), images.size(1), -1)
    var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])

std = torch.sqrt(var / ((len(train_loader.dataset) - 1) * 224 * 224))

# %%
# Mean, Standard deviation
# tensor([0.4631, 0.4483, 0.3237])
# tensor([0.2830, 0.2650, 0.2754])
print(mean)
print(std)
# %%
# Add transforms.
# Normalize with the statistics you found to your dataset constructor.
# This is in order to standardize (whiten) the input for better conditioned training
# and also in order to match what the pretrained model expects (it is trained on normalized Imagenet).
# Apply the same transform to the test dataset as well.
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_data = datasets.ImageFolder('butterflies/train', transform_norm)

# %%
# From the train dataset create two loaders:
# the loader used for optimizing hyperparameters (train_loader)
# and the loader used for validation (val_loader).
# Train Val split
train_size = len(train_data)
indices = list(range(train_size))
np.random.shuffle(indices)
split = int(np.floor((VAL_SPLIT) * train_size))
train_idx, valid_idx = indices[split:], indices[:split]
train_size = len(train_idx)
val_size = len(valid_idx)

train_set = torch.utils.data.Subset(train_data, train_idx)
val_set = torch.utils.data.Subset(train_data, valid_idx)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
## Part 2
# # %%
# # Freeze params
# for param in model.parameters():
#     param.requires_grad = False
#
# # %%
# # In you model architecture identify and delete the “classifier” part that maps “features” to scores of 1000 ImageNet classes.
# class_names = train_data.classes
# n_features = model.classifier[-1].in_features # Linear(in_features=4096, out_features=1000, bias=True)
# n_classes = len(class_names)

# # %%
# # Add a new “classifier” module that consists of one or more linear layers, with randomly initialized weights and outputs scores for 10 classes (our datasets).
# # If we construct Linear layers anew, their parameters are automatically randomly initialized and have the attribute requires_grad = True by default, i.e. will be trainable.
# # Consider using torch.nn.BatchNorm1d (after linear layers) or torch.nn.Dropout (after activations) inside your classifier block.
# added_layer = nn.Linear(n_features, n_classes)
# model.classifier[-1] = added_layer
# # print(model)
# %%
#Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class_names = train_data.classes
n_features = model.classifier[-1].in_features
n_classes = len(class_names)
fc = nn.Linear(n_features, n_classes)
init_weights(fc)
model.classifier[-1] = fc
model.eval()
# %%
lr = 0.0001
momentum = 0.9
step_size = 5
gamma = 0.1
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss = nn.CrossEntropyLoss()
loaders = {'train': train_loader, 'val': val_loader}
# model.cuda()
# conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Train the network and choose best parameters by cross-validation. Find a suitable learning rate as follows.
# First roughly determine the learning rate order by trying learning rates :
# 0.1
# 0.01
# 0.001
# 0.0001
# and comparing the training loss in 5 epochs.
def train_model_loss_compare(model, optimizer, loss_fce, loader, device, num_epochs=5):
    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        l = 0
        c = 0.0
        # Iterate over data.
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                energies, preds = torch.max(outputs, 1)
                loss = loss_fce(outputs, labels)
                loss.backward()
                optimizer.step()

            # statistics
            l += loss.item() * inputs.size(0)
            c += torch.sum(preds == labels.data)

        epoch_loss = l / len(loader.dataset)
        epoch_acc = c.double() / len(loader.dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
# %%
# train_model_loss_compare(model, optimizer, loss,train_loader, device, num_epochs=5)
# %%
# Knowing the rough value, select a grid of 5 learning rate values around it with which to perform full cross-validation.
lrs = [0.06, 0.08, 0.1, 0.12, 0.14]
# %%
# Evaluate validation accuracy after each epoch (as in lab2) and keep track of the parameter vector that achieves the best validation accuracy (saving the best so far).
# This way we automatically select the epoch at which it was the best to stop.
# Choose the learning rate that achieves the best validation accuracy. Apply regularization if needed (e.g. dropout or weight decay).
def train_model(model, loss_fce, optimizer, train_loader, val_loader, test_loader, device, num_epochs=10):
    val_losses = np.zeros(shape=(num_epochs,), dtype=float)
    val_accuracies = np.zeros(shape=(num_epochs,), dtype=float)
    train_losses = np.zeros(shape=(num_epochs,), dtype=float)
    train_accuracies = np.zeros(shape=(num_epochs,), dtype=float)
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = math.inf
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        l = 0.0
        c = 0.0

        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(x)
                energies, preds = torch.max(outputs, 1)
                loss = loss_fce(outputs, target)
                loss.backward()
                optimizer.step()

            # statistics
            l += loss.item() * x.size(0)
            c += torch.sum(preds == target.data).item()
            optimizer.param_groups[0]['lr'] += (0.0001)

        epoch_loss = l / len(train_loader.dataset)
        epoch_acc = c / len(train_loader.dataset)
        train_losses[epoch] = epoch_loss
        train_accuracies[epoch] = epoch_acc

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # validation
        with torch.set_grad_enabled(False):
            l = 0.0
            c = 0.0
            for x, target in val_loader:
                x = x.to(device)
                target = target.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(x)
                energies, preds = torch.max(outputs, 1)
                loss = loss_fce(outputs, target)

                l += loss.item() * x.size(0)
                c += torch.sum(preds == target.data).item()

            epoch_loss = l / len(val_loader.dataset)
            epoch_acc = c / len(val_loader.dataset)
            val_losses[epoch] = epoch_loss
            val_accuracies[epoch] = epoch_acc
            print(f'Val Epoch: {epoch} mean loss: {epoch_loss}, mean acc: {epoch_acc}')

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            elif epoch_acc == best_acc and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.eval()

    l = 0.0
    c = 0.0
    model.to(device)
    # Iterate over data.
    with torch.set_grad_enabled(False):
        for x, target in test_loader:
            x = x.to(device)
            target = target.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(x)
            energies, preds = torch.max(outputs, 1)
            loss = loss_fce(outputs, target)

            l += loss.item() * x.size(0)
            c += torch.sum(preds == target.data).item()

            if(torch.sum(preds == target.data).item() < len(target.data)):
                print(target.data == preds)
                print('Energies')
                print(energies)
                print('Preds')
                print(preds)
                print('Target')
                print(target)
        loss = l / len(test_loader.dataset)
        acc = c / len(test_loader.dataset)

        print('Model {} Loss: {:.4f} Acc: {:.4f}'.format(
                'Test', loss, acc))

    results = {'train_acc:': train_accuracies,'train_loss': train_losses, 'val_acc': val_accuracies, 'val_loss': val_losses}

    return model, results
# %%
test_data = datasets.ImageFolder('butterflies/train', transform_norm)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print('history_{:.2f}.pkl'.format(lr))
best_model, results = train_model(model, loss, optimizer, train_loader, val_loader, test_loader, device, num_epochs=1)
# best_model_save_name = 'Best_model + {:.4f}'.format(lr)
# torch.save(best_model.state_dict(), best_model_save_name)
# with open('history_{:.2f}.pkl'.format(lr), 'wb') as f:
#     pickle.dump(results, f)
# %%
# Report the full setup of learning that you used: base network, classifier architecture, optimizer, learning rate and other hyper-parameters. Report plots of training and validation metrics (loss, accuracy) versus epochs for the selected hyper-parameters.
# Report the final test classification accuracy.
test_data = datasets.ImageFolder('butterflies/train', transform_norm)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %%
# infile = open('/home/m/repo/pythonProject/zemiacko/dle/history_0.14.pkl','rb')
# history = pickle.load(infile)
# results = {'train_acc:': train_accuracies,'train_loss': train_losses, 'val_acc': val_accuracies, 'val_loss': val_losses}

# plt.plot(history['train_acc:'], 'b', label='Train acc')
# plt.plot(history['val_acc'], 'r', label='Validation acc')
# plt.title('Training and Validation acc lr = 0.14')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# %%
# %%
# If the network makes errors on the test data (we expect a few).
# For these cases display and report:
# 1) the input test image
# 2) its correct class label
# 3) the class labels and network confidence (predictive probabilities) of the top 3 network predictions (classes with highest predictive probability).