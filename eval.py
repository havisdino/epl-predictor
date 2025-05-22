from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule, Trainer

from utils.callbacks import CheckpointSaveEveryNSteps
from utils.data import batch_generator
from models.lightning_wrapper import LightningWrapper
from models.mlp import MLPWithAttention
from models.sr import SoftmaxRegressorWithAttention
from utils.config import Config


def main(config):
    assert hasattr(config, "checkpoint")
    
    if config.model_type == "mlp":
        model = MLPWithAttention(**vars(config.model_args))
    elif config.model_type == "sr":
        model = SoftmaxRegressorWithAttention(**vars(config.model_args))
    else:
        raise NotImplementedError()

    wrapper = LightningWrapper(model)
    
    config.trainer.devices = [0]
    
    trainer = Trainer(
        **vars(config.trainer),
        callbacks=[CheckpointSaveEveryNSteps(config.save_steps)]
    )
    
    def test_batch_generator():
        data_iter = batch_generator(config.data_files)
        for _ in range(1000):
            yield next(data_iter)
    
    trainer.test(wrapper, test_batch_generator(), ckpt_path=config.checkpoint)

if __name__ == "__main__":
    config = Config.from_yaml("config.yml")
    
    main(config)