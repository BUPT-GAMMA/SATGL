import os
import sys

current_dir = os.path.dirname(__file__)
relative_path = os.path.abspath(os.path.join(current_dir, '../'))
sys.path.insert(0, relative_path)

from satgl.config.config import Config

if __name__ == "__main__":
    # default config
    config = Config()

    # config with parameter
    config = Config(parameter_dict={"a": 1, "b": 2})
    print(config["a"], config["b"])

    # config with file
    config = Config(config_file_list=["./satgl/yaml/default.yaml"])