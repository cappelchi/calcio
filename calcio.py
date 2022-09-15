"""
Команды скрипта
 1. Развернуть окружение
 2. Обновить модель
 3. Получить предикт
 4. Обновить результаты

Установки
gensim=4.2.0
neptune-client
"""

import pandas as pd
import numpy as np
import neptune.new as neptune
import pickle
from gensim.models import KeyedVectors
import subprocess
from glob import glob
from tqdm import tqdm
from tensorflow.keras.models import load_model

pd.options.display.max_columns = 50
pd.options.display.max_rows = 100
print(pd.__version__)
print(np.__version__)


def run_bash(bashCommand: str, nameCommand=""):
    process = subprocess.Popen([bashCommand], shell=True)
    _, error = process.communicate()
    if error:
        print(f"{nameCommand} error:\n", error)


def get_credential(frmwork="neptune_team"):
    with open("credential.txt", "r") as container:
        for line in container:
            if frmwork in line:
                login, psw = line.split(" ")[1], line.split(" ")[2].split("\n")[0]
                return login, psw


def neptune_download(saved_name: str, local_path: str):
    '''
    :param saved_name:
    :param local_path:
    '''
    _, api_key = get_credential()
    project = neptune.init_project(name="scomesse/football", api_token=api_key)
    project[saved_name].download(local_path)
    project.stop()
    print('Ok...')


def set_environment():
    username, api_key = get_credential()
    print('Скачиваем граф матчей team_time_dict.pickle...')
    neptune_download('data/team_time_dict', './team_time_dict.pickle')

if __name__ == "__main__":
    set_environment()
