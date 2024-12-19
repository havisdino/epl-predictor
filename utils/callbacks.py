import os
from pytorch_lightning.callbacks import Callback


class CheckpointSaveEveryNSteps(Callback):
    def __init__(self, save_interval):
        super().__init__()
        self.save_interval = save_interval
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        step = trainer.global_step + 1
        if step % self.save_interval == 0:
            file_name = f"step-{step}.ckpt"
            save_path = os.path.join("ckpts", file_name)
            trainer.save_checkpoint(save_path)
            print(f"Checkpoint at step {step} saved")