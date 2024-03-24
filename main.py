# /usr/bin python

from pathlib import Path
from typing import List, Tuple
import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv, torch.nn.functional as F
from torchmetrics.classification import BinaryROC
import lightning as L


class CSVDataSet(data.Dataset):
    def __init__(self, path: Path):
        self.input_size = None
        with open(path) as f:
            self.data = []
            for line in f:
                if not line:
                    # Skip empty lines
                    continue
                *input_params, groud_truth = line.strip().split(",")
                if not input_params:
                    # Skip lines with less than 2 columns
                    continue

                input_params = list(map(float, input_params))

                # Input size has not been set yet, set it to the current line length
                if not self.input_size:
                    self.input_size = len(input_params)

                # Check that input size is consistent
                if len(input_params) != self.input_size:
                    raise ValueError(
                        f"Input params should have the same length as the first line: {self.input_size}"
                    )
                groud_truth = int(groud_truth)
                self.data.append((input_params, groud_truth))

    def input_dim(self):
        return self.input_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx][0], dtype=torch.float32),
            torch.tensor(self.data[idx][1], dtype=torch.int64),
        )


class BasicMLP(L.LightningModule):
    def __init__(self, input_dim: int, *hidden_dims: List[int]):
        super().__init__()
        self.model = nn.Sequential(
            *[
                nn.Linear(i, j)
                for i, j in zip(
                    [input_dim] + list(hidden_dims), list(hidden_dims) + [1]
                )
            ],
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        # training_step defines the train loop. It is independent of forward
        params, gt = batch
        gt = gt.to(torch.float32)
        pred = self.model(params)
        loss = F.mse_loss(pred, gt)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def roc_from_model(model_path, eval_set_path, batch_size, figure_path):
    roc = BinaryROC()
    dataset = CSVDataSet(eval_set_path)
    model = torch.load(model_path)

    for params, gt in data.DataLoader(dataset, batch_size=batch_size):
        pred = model(params)
        roc.update(pred, gt.unsqueeze(1))

    fig, axes = roc.plot(score=True)
    fig.savefig(figure_path)


def train(input_csv: Path, hidden_dims: List[int], batch_size: int, model_path: Path):
    dataset = CSVDataSet(input_csv)
    training_set, validation_set = data.random_split(
        dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)]
    )

    model = BasicMLP(dataset.input_dim(), *hidden_dims)
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(
        model,
        data.DataLoader(training_set, batch_size),
        data.DataLoader(validation_set),
    )
    # At the end of training, save the model and plot the ROC curve
    torch.save(model, model_path)
    roc_from_model(model_path, input_csv, batch_size, "ROC.png")


def test():
    pass


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Train, test, and evaluate a simple binary classifier, based on a MLP."
    )

    sub_parser = parser.add_subparsers(dest="subparser")
    # Arguments to this parser will be common to every subcommand, and available as members of args.
    parent_parser = ArgumentParser(add_help=False)

    # Either train, test, eval, roc, etc use batch size parameter!
    parser.add_argument("--batch-size", help="Batch size", type=int, default=32)

    train_parser = sub_parser.add_parser("train", parents=[parent_parser])
    train_parser.add_argument(
        "input_csv", help="Input CSV file for training", type=Path
    )
    train_parser.add_argument(
        "--hidden-dims",
        help="Hidden dimensions",
        type=lambda x: list(map(int, x.split(","))),
        default=[64, 64],
    )
    train_parser.set_defaults(func=train)

    test_parser = sub_parser.add_parser("test", parents=[parent_parser])
    test_parser.add_argument(
        "checkpoint",
        help="Path to the checkpoint",
        type=Path,
    )
    test_parser.set_defaults(func=test)

    roc_parser = sub_parser.add_parser("roc", parents=[parent_parser])
    roc_parser.add_argument(
        "model_path", help="Path to model file to evaluate ROC of", type=Path
    )
    roc_parser.add_argument(
        "eval_set_path", help="Path to the evaluation dataset", type=Path
    )
    roc_parser.add_argument(
        "--output",
        dest="figure_path",
        help="Path to figure output file",
        type=Path,
        default=Path("./ROC.png"),
    )
    roc_parser.set_defaults(func=roc_from_model)

    # Parse the arguments and call the appropriate function
    args = vars(parser.parse_args())
    func = args["func"]
    del args["subparser"]
    del args["func"]

    return func(**args)


if __name__ == "__main__":

    exit(main())
