import logging
from os.path import realpath
from utils import create_environment_config
from utils import unpack_tar
from utils import neptune_download
from utils import load_model

DATA_VERSION = 'data_221101/'
"""
Команды скрипта
 1. Развернуть окружение
 2. Обновить модель
 3. Получить предикт
 4. Обновить результаты
"""

def set_environment(destination_folder:str):
    # Развертывание окружения
    # 1. Загружаем все словари и эмбеддинги word2 vec
    # 2. Распаковываем word2vec
    # 3. Загружаем версию 1 модели NN для HOME предикшн
    destination_folder = realpath(destination_folder) + '/'

    logging.basicConfig(
        level=logging.INFO,
        filename=destination_folder + "set_environment.log",
        filemode="w",
    )
    env_dict = {
        DATA_VERSION + "team_GId_dict": "team_GId_dict.pickle",
        DATA_VERSION + "word2vec": "word2vec.wordvectors.tar.gz",
    }

    create_environment_config(
        {"destination_folder": destination_folder}
    )
    for cnt, env in enumerate(env_dict.items()):
        saved_name, file_name = env
        print(f"Скачиваем: {file_name}...{cnt + 1}/{len(env_dict)}")
        neptune_download(saved_name, destination_folder + file_name)
        if saved_name.split("/")[1] != "word2vec":
            create_environment_config(
                {saved_name.split("/")[1]: destination_folder + file_name}
            )

    create_environment_config(
        {"word2vec": unpack_tar(destination_folder + env_dict[VERSION + "word2vec"])}
    )
    model_num = 1
    model_type = "HOME"
    create_environment_config(
        {
            "tf_model": load_model(
                folder_name=destination_folder,
                model_num=model_num,
                model_type=model_type,
            ),
            "model_type": "HOME",
            "model_no": "1",
        }
    )

