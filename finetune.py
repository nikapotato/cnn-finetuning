# %%
import pickle

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
import time
import copy
import math
from torchvision.models.vgg import model_urls
# %% Normalise data
mean = torch.tensor([0.4631, 0.4483, 0.3237])
std = torch.tensor([0.2830, 0.2650, 0.2754])
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_data = datasets.ImageFolder('/home/m/repo/pythonProject/zemiacko/dle/butterflies', transform_norm)
# %% Split dataset
VAL_SPLIT = 1/4
BATCH_SIZE = 32
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
# %% Train
def train_model(model, loss_fce, optimizer, train_loader, val_loader, test_loader, device,num_epochs=20):
    val_losses = np.zeros(shape=(num_epochs,), dtype=float)
    val_accuracies = np.zeros(shape=(num_epochs,), dtype=float)
    train_losses = np.zeros(shape=(num_epochs,), dtype=float)
    train_accuracies = np.zeros(shape=(num_epochs,), dtype=float)
    model = model.to(device)
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
            optimizer.param_groups[0]['lr'] += (0.001)


        epoch_loss = l / len(train_loader.dataset)
        epoch_acc = c / len(train_loader.dataset)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        train_losses[epoch] = epoch_loss
        train_accuracies[epoch] = epoch_acc

        # validation
        with torch.set_grad_enabled(False):
            l = 0
            c = 0
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
            print(f'Val Epoch: {epoch} mean loss: {epoch_loss}, mean acc: {epoch_acc}')
            val_losses[epoch] = epoch_loss
            val_accuracies[epoch] = epoch_acc

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            elif epoch_acc == best_acc and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


    print('Best val Acc: {:4f}'.format(best_acc))
    results = {'train_acc:': train_accuracies, 'train_loss': train_losses, 'val_acc': val_accuracies,
               'val_loss': val_losses}
    # load best model weights
    model.load_state_dict(best_model_wts)
    model.eval()
    # l = 0.0
    # c = 0
    # # Iterate over data.
    # model.to(device)
    # with torch.set_grad_enabled(False):
    #     for x, target in test_loader:
    #         x = x.to(device)
    #         target = target.to(device)
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         outputs = model(x)
    #         energies, preds = torch.max(outputs, 1)
    #         loss = loss_fce(outputs, target)
    #
    #         l += loss.item() * x.size(0)
    #         c += torch.sum(preds == target.data).item()
    #
    #     loss = l / len(test_loader.dataset)
    #     acc = c / len(test_loader.dataset)
    #
    #     print('Model {} Loss: {:.4f} Acc: {:.4f}'.format(
    #     'Test', loss, acc))

    return model, results
# %% Test
test_data = datasets.ImageFolder('butterflies/train', transform_norm)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %% Load pretrained model
for k in model_urls.keys():
    model_urls[k] = model_urls[k].replace('https://', 'http://')

model = torchvision.models.vgg11(pretrained=True)
model.eval()
#Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class_names = train_data.classes
n_features = model.classifier[-1].in_features
n_classes = len(class_names)
fc = nn.Linear(n_features, n_classes)
init_weights(fc)
model.classifier[-1] = fc
model.eval()
# %%
lr = 0.00001
momentum = 0.9
step_size = 5

gamma = 0.1
loss = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# Decay LR by a factor of 0.1 every 10 epochs
#scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
# %%
best_model, results = train_model(model, loss, optimizer, train_loader, val_loader, test_loader, device)
best_model_save_name = f'Finetuned'
torch.save(best_model.state_dict(), best_model_save_name)
with open('history_finetune.pkl', 'wb') as f:
    pickle.dump(results, f)

# %%
# Test model
# test(best_model, test_loader, loss)

