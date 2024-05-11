import logging
import os
import uuid

class Logger(object):
    def __init__(self, log_file=False):
        self.log_file = log_file
        self.logger = logging.getLogger(str(uuid.uuid4()))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self.add_stream_handler()
        self.add_file_handler()
    
    def add_file_handler(self):
        if self.log_file is not False:
            if not os.path.exists(os.path.dirname(self.log_file)):
                os.makedirs(os.path.dirname(self.log_file))
            file_handler = logging.FileHandler(self.log_file)
            self.logger.addHandler(file_handler)

    def add_stream_handler(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)

    def debug(self, msg):
        self.logger.debug(msg)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def train_epoch_format(self, epoch, train_result):
        pre_fix = f"train"
        body = " | ".join([f"{key} : {'{:.6f}'.format(value)}" for key, value in train_result.items()])
        self.logger.info(f"{pre_fix} | {body}")
    
    def valid_epoch_format(self, epoch, train_result):
        pre_fix = f"valid"
        body = " | ".join([f"{key} : {'{:.6f}'.format(value)}" for key, value in train_result.items()])
        self.logger.info(f"{pre_fix} | {body}")

    def display_eval_results(self, eval_results: dict):
        max_key_len = max(len(str(key)) for key in eval_results.keys())
        max_val_len = max(len(str(val)) for val in eval_results.values())
        max_key_len = max(max_key_len, len('metric')) + 4
        max_val_len = max(max_val_len, len('result')) + 4

        header = '|' + 'metric'.ljust(max_key_len) + '|' + 'result'.ljust(max_val_len) + '|'
        self.logger.info('-' * len(header))
        self.logger.info(header)
        self.logger.info('-' * len(header))
        
        for key, val in eval_results.items():
            row = '|' + str(key).ljust(max_key_len) + '|' + str(val).ljust(max_val_len) + '|'
            self.logger.info(row)

        self.logger.info('-' * len(header))
