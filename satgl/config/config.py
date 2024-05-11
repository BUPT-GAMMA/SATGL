import yaml
import os
import re
import sys



default_yaml_path = os.path.join(os.path.dirname(__file__), "../yaml/default.yaml")

class Config(object):
    r"""
    Config class is a class that can be used to manage configuration files.
    Parameters:
    -----------
        config_file_list (list): List of configuration files to be loaded.
        parameter_dict (dict): Dictionary of parameters to be loaded.
    """

    def __init__(
        self, 
        config_file_list: list=None,
        parameter_dict: dict=None
        )-> None:
        cur_path = os.path.dirname(os.path.abspath(__file__))   
        self.default_config_file_list = [default_yaml_path]

        # copy parameter
        self.config_file_list = config_file_list
        self.parameter_dict = parameter_dict
        
        # build parameter
        self.default_dict = self._load_default_dict()
        self.file_dict = self._load_file_dict(config_file_list)
        self.parameter_dict = self._load_parameter_dict()
        self.cmd_dict = self._load_cmd_dict()
        self.config_dict = self._merge_dict()      
    
    def _load_parameter_dict(self) -> dict:
        if self.parameter_dict:
            return self.parameter_dict
        return dict()

    def _load_cmd_dict(self) -> dict:
        cmd_dict = dict()
        execution_environments = set(
            ["ipykernel_launcher", "colab"]
        )
        unrecognized_args = []

        if sys.argv[0] not in execution_environments:
            for arg in sys.argv[1:]:
                if arg.startswith('--') and len(arg.split('=')) == 2:
                    arg_key, arg_value = arg.split('=')
                    cmd_dict[arg_key[2:]] = arg_value
                else:
                    unrecognized_args.append(arg)
        if len(unrecognized_args) > 0:
            print("args [{}] be ignored".format(" | ".join(unrecognized_args)))
        return cmd_dict

    def _load_file_dict(self, config_file_list: list) -> dict:
        file_dict = dict()
        if config_file_list:
            for file in config_file_list:
                with open(file, "r", encoding="utf-8") as f:
                    file_dict.update(
                        yaml.load(f.read(), Loader=yaml.FullLoader)
                    )
        return file_dict

    def _load_default_dict(self) -> dict:
        return self._load_file_dict(self.default_config_file_list)

    def _merge_dict(self) -> dict:
        config_dict = self.default_dict
        config_dict.update(self.file_dict)
        config_dict.update(self.parameter_dict)
        config_dict.update(self.cmd_dict)
        return config_dict
        
    def __setitem__(self, key, value) -> None:
        if not isinstance(key, str):
            raise TypeError("key must be str")
        self.config_dict[key] = value
    
    def __getattr__(self, item) -> any:
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            raise AttributeError("No such attribute: {}".format(item))
    
    def __getitem__(self, item) -> any:
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            raise KeyError("No such key: {}".format(item))

    def __str__(self) -> str:
        return str(self.config_dict)
    
    def __repr__(self) -> str:
        return self.__str__()
    