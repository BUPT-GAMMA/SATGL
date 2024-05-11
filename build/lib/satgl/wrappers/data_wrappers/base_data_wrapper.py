import os

from satgl.loggers.logger import Logger

class DataWrapper(object):
    def __init__(self, root_dir, log_file):
        self.root_dir = root_dir
        self.logger = Logger(log_file)
        self.process()

    
    def get_dataset(self, dataset_path: str, label_path: str):
        raise NotImplementedError   
        

    def get_dataloader(self, dataset: object):
        raise NotImplementedError

    def process(self):
        # get data directories
        train_data_dir = os.path.join(self.root_dir, "train")
        valid_data_dir = os.path.join(self.root_dir, "valid")
        test_data_dir = os.path.join(self.root_dir, "test")
        
        # get label paths
        train_label_path = os.path.join(self.root_dir, "label", "train.csv")
        valid_label_path = os.path.join(self.root_dir, "label", "valid.csv")
        test_label_path = os.path.join(self.root_dir, "label", "test.csv")

        # get datasets
        self.logger.info("processing train dataset ...")
        self.train_dataset = self.get_dataset(train_data_dir, train_label_path)
        self.logger.info("processing valid dataset ...")
        self.valid_dataset = self.get_dataset(valid_data_dir, valid_label_path)
        self.logger.info("processing test dataset ...")
        self.test_dataset = self.get_dataset(test_data_dir, test_label_path)

        # get dataloaders
        self.train_dataloader = self.get_dataloader(self.train_dataset, shuffle=True)
        self.valid_dataloader = self.get_dataloader(self.valid_dataset)
        self.test_dataloader = self.get_dataloader(self.test_dataset)

