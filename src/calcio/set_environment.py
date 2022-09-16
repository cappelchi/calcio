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

from utils import create_environment_config
from utils import unpack_tar
from utils import neptune_download
from utils import load_model

pd.options.display.max_columns = 50
pd.options.display.max_rows = 100
print(pd.__version__)
print(np.__version__)


def set_environment(destination_folder = "../../"):
    # Развертывание окружения
    # 1. Загружаем все словари и эмбеддинги word2 vec
    # 2. Распаковываем word2vec
    # 3. Загружаем версию 1 модели NN для HOME предикшн
    logging.basicConfig(
        level=logging.INFO, filename="../../set_environment.log", filemode="w"
    )
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
        {'destination_folder': destination_folder}
    )
    for cnt, env in enumerate(env_dict.items()):
        saved_name, file_name = env
        print(f"Скачиваем: {file_name}...{cnt + 1}/{len(env_dict)}")
        neptune_download(saved_name, destination_folder + file_name)
        if saved_name != "data/word2vec_220811":
            create_environment_config(
                {saved_name.split("/")[1]: destination_folder + file_name}
            )

    create_environment_config(
        {"word2vec": unpack_tar(destination_folder + env_dict["data/word2vec_220811"])}
    )
    model_num = 1
    model_type = "HOME"
    create_environment_config(
        {
            "tf_model": load_model(
                folder_name = destination_folder,
                model_num=model_num,
                model_type=model_type
            ),
            'model_type':'HOME',
            'model_no':'1'
        }
    )



#if __name__ == "__main__":
#    set_environment(local_folder="./")
