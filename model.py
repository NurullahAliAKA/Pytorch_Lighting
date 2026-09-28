
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from data import MyDataModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class MyModel(L.LightningModule):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  # __init__ parametrelerini otomatik kaydeder

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

    # --- Eğitim adımı ---
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # --- Doğrulama adımı ---
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    # --- Test adımı ---
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss)

    # --- Optimizer tanımı ---
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

if __name__ == "__main__":
    model = MyModel(input_dim=784, hidden_dim=128, output_dim=10)
    # TensorBoard
    logger = TensorBoardLogger("logs/", name="my_model")
    # veya Weights & Biases
    # logger = WandbLogger(project="my-project")

    trainer = L.Trainer(max_epochs=10,
                        logger=logger,
                        )
    trainer.fit(model,datamodule=MyDataModule())
    trainer.test(model,datamodule=MyDataModule())