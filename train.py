import torch
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
from torch.utils.data import DataLoader, TensorDataset
from net import MyCNN
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = []
labels = []


folder_cats = "dataset/training_set/cats"
folder4_dogs = "dataset/training_set/dogs"


def add_images(folder, label: int):
    for i in os.listdir(folder):

        if i.endswith(".jpg") or i.endswith(".png") or i.endswith("jpeg"):
            image = cv2.imread(os.path.join(folder, i))
            image = cv2.resize(image, (64, 64))
            data.append(image)
            labels.append(label)


add_images(folder_cats, 0)
add_images(folder4_dogs, 1)


# print(len(data))

data = np.array(data)
labels = np.array(labels)
data = data.transpose(0, 3, 1, 2)

data = torch.from_numpy(data / 255).float()
labels = torch.from_numpy(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

network = MyCNN()
network.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)

train_losses = []
validation_losses = []
train_accuracies = []
validation_accuracies = []
highest_accuracy = 0


def train():
    losses = []
    correct = 0
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target.long())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        predictions = []

        for i in output:
            label = i.cpu().detach().numpy().argmax()
            predictions.append(label)

        for i in range(0, len(predictions)):
            if target[i] == predictions[i]:
                correct += 1
    train_accuracies.append((correct / len(train_loader.dataset)) * 100)
    train_losses.append(sum(losses) / len(losses))

    print(f"Training loss: {sum(losses) / len(losses)}")
    print(f"Training accuracy: {(correct / len(train_loader.dataset)) * 100}")


def test(epoch):
    losses = []
    correct = 0
    global highest_accuracy
    network.eval()

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            loss = criterion(output, target.long())
            losses.append(loss.item())

            predictions = []
            for i in output:
                label = i.cpu().detach().numpy().argmax()
                predictions.append(label)

            for i in range(0, len(predictions)):
                if target[i] == predictions[i]:
                    correct += 1
        validation_accuracies.append((correct / len(test_loader.dataset)) * 100)
        validation_losses.append(sum(losses) / len(losses))
        print(f"Validation loss: {sum(losses) / len(losses)}")
        print(f"Validation accuracy: {(correct / len(test_loader.dataset)) * 100}")

        if (correct / len(test_loader.dataset)) * 100 > highest_accuracy:
            highest_accuracy = (correct / len(test_loader.dataset)) * 100
            torch.save(network, "models/" + str(epoch + 1) + '-' + str((correct / len(test_loader.dataset)) * 100))


epochs = 20
for epoch in range(epochs):
    print(f"Training epoch: {epoch + 1}")
    train()
    test(epoch)

fig1 = plt.figure()
plt.plot(range(1, epochs + 1), train_losses, color="blue")
plt.plot(range(1, epochs + 1), validation_losses, color="red")

plt.legend(["Training Loss", "Validation Loss"], loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")

fig2 = plt.figure()
plt.plot(range(1, epochs + 1), train_accuracies, color="blue")
plt.plot(range(1, epochs + 1), validation_accuracies, color="red")

plt.legend(["Training Accuracy", "Validation Accuracy"], loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.show()
