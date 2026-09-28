import pytorch_lightning as pl
import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader,random_split

class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32),
            v2.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        # İndirme gibi tek seferlik, dağıtık ortamda sadece 1 kez çalışan işlemler
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Her GPU/process'te çalışır, veri setlerini oluşturur
        full_train = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        self.train_set, self.val_set = random_split(full_train, [55000, 5000])
        self.test_set = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4)

if __name__ == "__main__":
    dm = MyDataModule()
