"""
Команды скрипта
 1. Развернуть окружение
 2. Обновить модель
 3. Получить предикт
 4. Обновить результаты
"""

import pandas as pd
import numpy as np
import neptune.new as neptune
import os
import tarfile
import logging
import yaml
import pickle
from gensim.models import KeyedVectors
from glob import glob
from tqdm import tqdm
from tensorflow.keras.models import load_model

pd.options.display.max_columns = 50
pd.options.display.max_rows = 100
print(pd.__version__)
print(np.__version__)


def get_credential(frmwork="neptune_team"):
    with open("../../credential.txt", "r") as container:
        for line in container:
            if frmwork in line:
                login, psw = line.split(" ")[1], line.split(" ")[2].split("\n")[0]
                return login, psw


def unpack_tar(file_path: str) -> list:
    """
    :param file_path:
    :return:list
    """
    untar_list = []
    if file_path[-6:] == "tar.gz":
        if os.path.isfile(file_path):
            print(f"Распаковывание {file_path}")
            with tarfile.open(file_path, "r:gz") as targz:
                untar_list = [nms for nms in targz.getnames()]
                targz.extractall()
            logging.info(f"Unpacked: {file_path}")
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Aрхив удалён: {file_path}")
                logging.info(f"Archive deleted: {file_path}")
            else:
                print("Не получилось удалить архив после распаковки")
        else:
            print(f"Файл {file_path} не существует")
    else:
        print(f"Файл {file_path} не имеет расширение tar.gz")
    return untar_list


def neptune_download(saved_name: str, local_path: str):
    """
    :param saved_name:
    :param local_path:
    """
    try:
        _, api_key = get_credential()
        project = neptune.init_project(name="scomesse/football", api_token=api_key)
        project[saved_name].download(local_path)
        project.stop()
        print("Ok...")
        logging.info(f"Downloaded: {local_path}")
    except Exception:
        logging.error(f"Downloaded neptune: {local_path}")


def load_model(folder_name: str, model_type="HOME", model_num=1) -> list:
    """
    :param folder_name:
    :param model_num:
    :param model_type:
    """
    path_to_model = folder_name + "model.tar.gz"
    _, api_key = get_credential()
    neptune_model = f"FOOT-" + model_type
    neptune_model_version = f"FOOT-" + model_type + "-" + str(model_num)
    model_version = neptune.init_model_version(
        project="scomesse/football",
        model=neptune_model,
        api_token=api_key,
        version=neptune_model_version,
    )
    try:
        print(f"Загружаем модель {model_type} n.{model_num}")
        model_version["model"].download(path_to_model)
        model_version.stop()
        logging.info(f"Downloaded: {path_to_model}")
    except Exception:
        logging.error(f"Downloaded neptune: {path_to_model}")
    print(f"Распаковываем модель {model_type} n.{model_num}")
    return unpack_tar(path_to_model)


def create_environment_config(config_dict: dict):
    if os.path.isfile("../../config.yaml"):
        with open("../../config.yaml", "r") as conf:
            current_config = yaml.load(conf, Loader=yaml.SafeLoader)
        for key, value in config_dict.items():
            current_config[key] = value
        with open("../../config.yaml", "w") as conf:
            yaml.dump(current_config, conf)
    else:
        with open("../../config.yaml", "w") as conf:
            yaml.dump(config_dict, conf)


def set_environment(local_folder="./"):
    # Развертывание окружения
    # 1. Загружаем все словари и эмбеддинги word2 vec
    # 2. Распаковываем word2vec
    # 3. Загружаем версию 1 модели NN для HOME предикшн
    logging.basicConfig(
        level=logging.INFO, filename="../../set_environment.log", filemode="w"
    )
    config_dict = {}
    env_dict = {
        "data/team_time_dict": "team_time_dict.pickle",
        "data/team_league_dict": "team_league_dict.pickle",
        "data/season_dict": "season_dict.pickle",
        "data/idx_home_current_dict": "idx_home_current_dict.pickle",
        "data/idx_away_current_dict": "idx_away_current_dict.pickle",
        "data/idx_home_dict": "idx_home_dict.pickle",
        "data/idx_away_dict": "idx_away_dict.pickle",
        "data/word2vec_220811": "w2v_model.tar.gz",
    }

    create_environment_config(
        {'main_folder': local_folder}
    )
    for cnt, env in enumerate(env_dict.items()):
        saved_name, file_name = env
        print(f"Скачиваем: {file_name}...{cnt + 1}/{len(env_dict)}")
        neptune_download(saved_name, local_folder + file_name)
        if saved_name != "data/word2vec_220811":
            create_environment_config(
                {saved_name.split("/")[1]: local_folder + file_name}
            )

    create_environment_config(
        {"word2vec": unpack_tar(local_folder + env_dict["data/word2vec_220811"])}
    )
    model_num = 1
    model_type = "HOME"
    create_environment_config(
        {
            "tf_model": load_model(
                folder_name="./", model_num=model_num, model_type=model_type
            ),
            'model_type':'HOME',
            'model_no':'1'
        }
    )



if __name__ == "__main__":
    set_environment(local_folder="./")
