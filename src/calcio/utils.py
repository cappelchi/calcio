import os
import tarfile
import logging
import pickle
import yaml
import neptune.new as neptune


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


def create_environment_config(config_dict: dict, destination_folder="../../"):
    config_path = destination_folder + "config.yaml"
    if os.path.isfile(config_path):
        with open(config_path, "r") as conf:
            current_config = yaml.load(conf, Loader=yaml.SafeLoader)
        for key, value in config_dict.items():
            current_config[key] = value
        with open(config_path, "w") as conf:
            yaml.dump(current_config, conf)
    else:
        with open(config_path, "w") as conf:
            yaml.dump(config_dict, conf)


def unpack_tar(file_path: str) -> list:
    """
    :param file_path:
    :return:list
    """
    archive_folder = file_path.replace(file_path.split('/')[-1], '')
    untar_list = []
    if file_path[-6:] == "tar.gz":
        if os.path.isfile(file_path):
            print(f"Распаковывание {file_path}")
            with tarfile.open(file_path, "r:gz") as targz:
                untar_list = [archive_folder + '/'.join(nms.split('/')[1:]) for nms in targz.getnames()]
                targz.extractall(path = archive_folder)
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


def neptune_download(saved_name: str, file_path: str):
    """
    :param saved_name:
    :param local_path:
    """
    try:
        _, api_key = get_credential()
        project = neptune.init_project(name="scomesse/football", api_token=api_key)
        project[saved_name].download(file_path)
        project.stop()
        print("Ok...")
        logging.info(f"Downloaded: {file_path}")
    except Exception:
        logging.error(f"Downloaded neptune: {file_path}")


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

def load_dicts(main_folder):
    dicts_list = []
    dicts_names_list = ['team_time_dict', 'team_league_dict', 'season_dict', 'idx_home_current_dict',
                  'idx_away_current_dict', 'idx_home_dict', 'idx_away_dict']
    for single_dict in dicts_names_list:
        with open(main_folder + single_dict + '.pickle', 'rb') as pkl:
            dicts_list.append(pickle.load(pkl))

    return dicts_list


