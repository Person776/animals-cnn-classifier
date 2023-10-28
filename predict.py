import os

import numpy as np
import torch
import cv2
from natsort import natsorted
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "models/10-80.25"
model = torch.load(model_path)
model.to(device)
model.eval()

predicted_labels = []


def classify(prediction):

    label = prediction[0].cpu().detach().numpy().argmax()
    predicted_labels.append(label)


image_files = os.listdir("predict")
image_list = natsorted(f for f in image_files if f.endswith('.png') or f.endswith('.jpg') or f.endswith("jpeg"))

images = []

for i in image_list:
    img = cv2.imread("predict/" + i)
    img = cv2.resize(img, (64, 64))
    images.append(img)

images = np.array(images)
images = images / 255
images = images.transpose(0, 3, 1, 2)

predictions = []
for image in images:
    image = torch.from_numpy(np.array([image])).float().to(device)
    predictions.append(model(image))

for i in predictions:
    classify(i)

print(predicted_labels)

if os.path.exists("predict/labels.csv"):
    os.remove("predict/labels.csv")

f = open("predict/labels.csv", "a")
writer = csv.writer(f)

for i in range(len(predicted_labels)):
    animals = {0: "Cat", 1: "Dog"}
    row = [image_list[i], predicted_labels[i],
           animals[predicted_labels[i]]]
    writer.writerow(row)

f.close()
