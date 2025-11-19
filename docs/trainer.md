```
from smle.utils import set_seed
from smle.trainer import Trainer
from smle import smle


import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from sklearn.metrics import confusion_matrix, classification_report


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


@smle
def main(args):


    set_seed(args["training"]["seed"])


    df_train = pd.read_csv(args["dataset"]["train_file"])
    df_test = pd.read_csv(args["dataset"]["test_file"])


    X_train = df_train.iloc[:, 1:].values
    y_train = df_train.iloc[:, 0].values


    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values


    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)


    train_loader = DataLoader(train_dataset, batch_size=args["training"]["train_batch"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args["training"]["test_batch"], shuffle=False)


    device = args["training"]["device"]


    model = CustomMLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=float(args["training"]["learning_rate"]),
                            weight_decay=float(args["training"]["weight_decay"]))


    trainer = Trainer(model, loss_fn, optimizer, device=args["training"]["device"])


    model = trainer.fit(train_loader=train_loader,
                        epochs=args["training"]["epochs"],
                        export_dir=args["training"]["export_dir"])


    predictions = []
    grounds = []


    model.eval()
    for data,labels in test_loader:


        data = data.to(device)


        probs = model(data).squeeze()
        preds = torch.argmax(probs, axis=1)


        predictions.extend(preds.detach().cpu().tolist())
        grounds.extend(labels.detach().cpu().tolist())


    print(f"{classification_report(grounds, predictions)}")
    print(f"{confusion_matrix(grounds, predictions)}")


if __name__ == "__main__":
    main()
```