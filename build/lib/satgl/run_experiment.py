from satgl.trainer.trainer import Trainer
from satgl.config.config import Config
from satgl.wrappers.get_data import get_data
from satgl.trainer.get_trainer import get_trainer
from satgl.wrappers.get_model import get_model



def run_experiment(config: Config):
    data = get_data(config)
    trainer = get_trainer(config)
    model = get_model(config)
    trainer.train(model, data.train_dataloader, data.valid_dataloader, data.test_dataloader)