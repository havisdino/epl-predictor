import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class LightningWrapper(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        self._acc_train = 0
        self._count = 0
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
    
    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, inputs):
        logits = self.model(inputs)        
        target = inputs["next_match_result"]
        
        loss = F.cross_entropy(logits, target, label_smoothing=0.05)
        
        acc_train_new = (logits.argmax(-1) == target) / target.numel()
        self._acc_train = (self._acc_train * self._count + acc_train_new) / (self._count + 1)
        self._count += 1
        
        self.log("loss", loss.detach().item(), prog_bar=True, sync_dist=True)

        return loss
    
    def on_before_zero_grad(self, optimizer):
        super().on_before_zero_grad(optimizer)
        
        self.log("acc_train", self._acc_train.item(), prog_bar=True, sync_dist=True)
        self._acc_train = 0
        self._count = 0