from torchmetrics import Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from data import MyDataModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


class MyModel(L.LightningModule):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10,lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  # __init__ parametrelerini otomatik kaydeder
        #self.model=model() #--- oluşturulan model hariciden eklme
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.val_acc = Accuracy(task="multiclass",num_classes=output_dim)
        self.test_acc = Accuracy(task="multiclass",num_classes=output_dim)

    def forward(self, x):
        #x=self.model(x) #-- model kullanımı
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

    #--- logits ve loss hesaplma--- her adımda kod tekrarını önlemek için bir method kullandık.
    def _data_setup(self,batch,batch_idx):#
        """
        x=Girdi verisi (örneğin bir görüntü tensor'ü, şekli: [batch_size, kanal, yükseklik, genişlik])
        y=Gerçek etiketler (örneğin sınıf numaraları, şekli: [batch_size])
        :param batch:DataLoader'dan gelen tek bir veri grubu. Genelde (girdi, etiket) tuple'ı şeklindedir, yani (x, y).
        :param batch_idx:O epoch içinde bu batch'in kaçıncı sırada olduğunu gösteren indeks (0, 1,]
        :return:En kritik satır. Lightning, bu dönen loss değerini alıp arka planda otomatik olarak geriyayılım başlatır.
        """
        x, y = batch #-- gelen verinin batch lerini al.
        x = x.view(x.size(0), -1)# flatten (düzleştirme) cnn modellerde kullanma
        logits = self(x) # forward methodu ile ham çıktı üret
        loss = F.cross_entropy(logits, y) # ham çıktıları ve gerçek verilerin hatasını hesaplar. içerisinde Softmax otomatik uygulanır.
        return loss, logits,y

    # --- Eğitim adımı ---
    def training_step(self, batch, batch_idx):
        loss, logits,y = self._data_setup(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # --- Doğrulama adımı ---
    def validation_step(self, batch, batch_idx):
        loss,logits,y=self._data_setup(batch, batch_idx)
        acc = self.val_acc.update(logits,y) ## sadece state günceller, anlık değer dönmez. Bu nedenle update önemli.
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    # --- Test adımı ---
    def test_step(self, batch, batch_idx):
        loss,logits,y=self._data_setup(batch, batch_idx)
        acc = self.test_acc.update(logits,y)## sadece state günceller, anlık değer dönmez. Bu nedenle update önemli.
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    # --- Optimizer tanımı ---
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler ile öğrenme oranını her 10 adımda bir düşürür. modeli dengeli bir öğrenme oranında tutar.
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