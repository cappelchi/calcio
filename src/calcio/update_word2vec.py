from utils import create_environment_config


def update_word2vec(saved_name = '', file_name = ''):
    # TODO: Дописать обновление word2vec модели
    print(f"Скачиваем: {file_name}...{cnt + 1}/{len(env_dict)}")
    neptune_download(saved_name, file_name)
    create_environment_config(
        {"word2vec": unpack_tar('')}
    )