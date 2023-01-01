import torch
import timm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy, confusion_matrix


class LitResnet(pl.LightningModule):
    def __init__(self, num_classes=10, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)

        if stage:
            self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True)
            self.log(f"{stage}/acc", acc, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": y}     

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        test_dict = self.evaluate(batch, "test")
        return test_dict

    def test_epoch_end(self, outputs):
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        confusion_mat = confusion_matrix(preds, targets, task="multiclass", num_classes=self.num_classes)      

        df_cm = pd.DataFrame(confusion_mat.cpu().numpy(), index = range(6), columns=range(6))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt="d").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)
        
        confusion_mat = confusion_mat.type(torch.float32)
        self.log("test/confusion_matrix/00", confusion_mat[0][0], on_epoch=True)
        self.log("test/confusion_matrix/01", confusion_mat[0][1], on_epoch=True)
        self.log("test/confusion_matrix/02", confusion_mat[0][2], on_epoch=True)
        self.log("test/confusion_matrix/03", confusion_mat[0][3], on_epoch=True)
        self.log("test/confusion_matrix/04", confusion_mat[0][4], on_epoch=True)
        self.log("test/confusion_matrix/05", confusion_mat[0][5], on_epoch=True)

        self.log("test/confusion_matrix/10", confusion_mat[1][0], on_epoch=True)
        self.log("test/confusion_matrix/11", confusion_mat[1][1], on_epoch=True)
        self.log("test/confusion_matrix/12", confusion_mat[1][2], on_epoch=True)
        self.log("test/confusion_matrix/13", confusion_mat[1][3], on_epoch=True)
        self.log("test/confusion_matrix/14", confusion_mat[1][4], on_epoch=True)
        self.log("test/confusion_matrix/15", confusion_mat[1][5], on_epoch=True)

        self.log("test/confusion_matrix/20", confusion_mat[2][0], on_epoch=True)
        self.log("test/confusion_matrix/21", confusion_mat[2][1], on_epoch=True)
        self.log("test/confusion_matrix/22", confusion_mat[2][2], on_epoch=True)
        self.log("test/confusion_matrix/23", confusion_mat[2][3], on_epoch=True)
        self.log("test/confusion_matrix/24", confusion_mat[2][4], on_epoch=True)
        self.log("test/confusion_matrix/25", confusion_mat[2][5], on_epoch=True)

        self.log("test/confusion_matrix/30", confusion_mat[3][0], on_epoch=True)
        self.log("test/confusion_matrix/31", confusion_mat[3][1], on_epoch=True)
        self.log("test/confusion_matrix/32", confusion_mat[3][2], on_epoch=True)
        self.log("test/confusion_matrix/33", confusion_mat[3][3], on_epoch=True)
        self.log("test/confusion_matrix/34", confusion_mat[3][4], on_epoch=True)
        self.log("test/confusion_matrix/35", confusion_mat[3][5], on_epoch=True)

        self.log("test/confusion_matrix/40", confusion_mat[4][0], on_epoch=True)
        self.log("test/confusion_matrix/41", confusion_mat[4][1], on_epoch=True)
        self.log("test/confusion_matrix/42", confusion_mat[4][2], on_epoch=True)
        self.log("test/confusion_matrix/43", confusion_mat[4][3], on_epoch=True)
        self.log("test/confusion_matrix/44", confusion_mat[4][4], on_epoch=True)
        self.log("test/confusion_matrix/45", confusion_mat[4][5], on_epoch=True)

        self.log("test/confusion_matrix/50", confusion_mat[5][0], on_epoch=True)
        self.log("test/confusion_matrix/51", confusion_mat[5][1], on_epoch=True)
        self.log("test/confusion_matrix/52", confusion_mat[5][2], on_epoch=True)
        self.log("test/confusion_matrix/53", confusion_mat[5][3], on_epoch=True)
        self.log("test/confusion_matrix/54", confusion_mat[5][4], on_epoch=True)
        self.log("test/confusion_matrix/55", confusion_mat[5][5], on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        return {"optimizer": optimizer}