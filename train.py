from efficientNetModel import *
from datatset import *
from config import *
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm

args = cArgs()
print('device: ', args.device)


def train_fn(model, train_loader, optimizer, loss_fn):
    loop = tqdm(train_loader)
    total_loss = []
    for batch, (image, label) in enumerate(loop):
        image = image.to(args.device)
        label = label.float()

        predictions = model(image).squeeze(1)

        loss = loss_fn(predictions, label)
        total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Average training loss per batch: {sum(total_loss)/len(total_loss)}')
    return sum(total_loss)/len(total_loss)


def main():
    train_losses = []
    val_losses = []

    # load training and validation data
    train_data = DogsVsWolvesDataset(args.train_dir)
    val_data = DogsVsWolvesDataset(args.val_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False)

    # create and load model
    model = EfficientNet('b0', num_classes=1).to(args.device)  # num classes = 1 since we do binary classification, meaning that we really just need one class..
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    if args.load_checkpoint:
        load_checkpoint(model, optimizer, args.lr)

    for epoch in range(args.epochs):
        print(f'Training epoch {epoch+1}..\n')
        epoch_loss = train_fn(model, train_loader, optimizer, loss_fn)
        train_losses.append(epoch_loss)

        if args.save_model and (epoch+1) % args.save_frequency == 0:
            save_checkpoint(model, optimizer, args.save_path)

        if (epoch+1) % args.validation_frequency == 0:
            print('Validating the Accuray..')
            accuracy = getAccuracy(model, val_loader, args.device)
            print(f'Accuracy after epoch {epoch}: {accuracy}')
            val_losses.append(accuracy)


if __name__ == '__main__':
    main()