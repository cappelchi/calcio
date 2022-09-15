
'''
Команды скрипта
 1. Развернуть окружение
 2. Обновить модель
 3. Получить предикт
 4. Обновить результаты

Установки
gensim=4.2.0
neptune-client
'''

import pandas as pd
import numpy as np
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

def get_credential(frmwork = 'neptune_team'):
    with open('credential.txt', 'r') as container:
        for line in container:
            if frmwork in line:
                login, psw = line.split(' ')[1], line.split(' ')[2].split('\n')[0]
                return login, psw

username, api_key = get_credential()
def set_environment():
    pass


if __name__ == '__main__':
    print('Start here')