import traceback

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from smle.utils import set_seed
from smle.utils.logger import Logger
from smle.utils.args import ArgParser

class CustomMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self._in_h1 = nn.Linear(784, 128)
        self._h1_h2 = nn.Linear(128, 64)
        self._h2_out = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self._in_h1(x))
        x = F.relu(self._h1_h2(x))
        x = self._h2_out(x)
        return x

class CustomDataset(Dataset):

    def __init__(self, X, y):

        super().__init__()
        self._X = X
        self._y = y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, index):

        x = torch.tensor(self._X[index], dtype=torch.float)
        y = torch.tensor(self._y[index], dtype=torch.long) # Cross Entropy requires long

        return x, y

def main(args, logger):

    df = pd.read_csv(args["dataset"]["dir"])

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=args["training"]["seed"])


    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args["training"]["train_batch"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args["training"]["test_batch"], shuffle=False)

    device = args["training"]["device"]

    model = CustomMLP()
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=float(args["training"]["learning_rate"]),
                            weight_decay=float(args["training"]["weight_decay"]))

    model.train()
    for epoch in range(args["training"]["epochs"]):

        logger.log(f"Epoch {epoch+1} [STARTED]")
        mean_loss = 0

        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            logit = model(data).squeeze()

            loss = loss_fn(logit, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            mean_loss += loss.item()

        mean_loss /= len(train_dataset)

        logger.log(f"Epoch {epoch+1} [ENDED]. Loss: {mean_loss:.4f}")

    predictions = []
    grounds = []

    model.eval()
    for data,labels in test_loader:

        data = data.to(device)

        probs = model(data).squeeze()
        preds = torch.argmax(probs, axis=1)

        predictions.extend(preds.detach().cpu().tolist())
        grounds.extend(labels.detach().cpu().tolist())

    logger.log(f"\n{classification_report(grounds, predictions)}")
    logger.log(f"\n{confusion_matrix(grounds, predictions)}")
if __name__ == "__main__":

    parser = ArgParser(prog="LLM finetuning",
                       description="Fine-tune LLM models with YAML configuration")

    logger = Logger(parser)

    set_seed(parser.args["training"]["seed"])

    try:
        main(parser.args, logger)
    except:
        logger.log(traceback.format_exc())

