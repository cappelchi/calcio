import os
import yaml


def get_credential(frmwork="neptune_team"):
    with open("../../credential.txt", "r") as container:
        for line in container:
            if frmwork in line:
                login, psw = line.split(" ")[1], line.split(" ")[2].split("\n")[0]
                return login, psw


def get_environment_config(config_path="../../config.yaml") -> dict:
    if os.path.isfile(config_path):
        with open(config_path, "r") as conf:
            current_config = yaml.load(conf, Loader=yaml.SafeLoader)
        return current_config
    else:
        print("Нет конфига, запустите развёртывание оеружения")
        return {}


def set_config(config_dict, config_path="../../config.yaml") -> dict:
    if os.path.isfile(config_path):
        with open(config_path, "r") as conf:
            current_config = yaml.load(conf, Loader=yaml.SafeLoader)
        for key, value in config_dict.items():
            current_config[key] = value
        with open(config_path, "w") as conf:
            yaml.dump(current_config, conf)
    else:
        print("Нет конфига, запустите развёртывание оеружения")
