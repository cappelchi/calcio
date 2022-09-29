import os
import tarfile
import logging
import pickle
import functools
import pandas as pd
import yaml
import neptune.new as neptune
from tqdm import tqdm


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
    archive_folder = file_path.replace(file_path.split("/")[-1], "")
    untar_list = []
    if file_path[-6:] == "tar.gz":
        if os.path.isfile(file_path):
            print(f"Распаковывание {file_path}")
            with tarfile.open(file_path, "r:gz") as targz:
                untar_list = [
                    archive_folder + "/".join(nms.split("/")[1:])
                    for nms in targz.getnames()
                ]
                targz.extractall(path=archive_folder)
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
    dicts_names_list = [
        "team_time_dict",
        "team_league_dict",
        "season_dict",
        "idx_home_current_dict",
        "idx_away_current_dict",
        "idx_home_dict",
        "idx_away_dict",
    ]
    for single_dict in dicts_names_list:
        with open(main_folder + single_dict + ".pickle", "rb") as pkl:
            dicts_list.append(pickle.load(pkl))

    return dicts_list


def load_dataframe(folder: str, start_date="", end_date="") -> pd.DataFrame:
    """
    :param folder:
    :param start_date:
    :param end_date:
    :return pandas Dataframe:
    """
    data_csv_list = [
        folder + str(dd).replace("-", "") + ".csv"
        for dd in pd.date_range(start=start_date, end=end_date).date
    ]

    data_df = pd.concat(
        map(functools.partial(pd.read_csv, sep=";", compression=None), data_csv_list),
        ignore_index=True,
    )
    data_df[["date", "times_ext"]] = data_df.Time.str.split(expand=True)
    data_df = data_df.drop(["Time", "times_ext"], axis="columns")
    data_df.date = pd.to_datetime(data_df["date"], dayfirst=True)
    data_df = data_df.sort_values(by="date").reset_index(drop=True)
    data_df.HomeId = data_df.HomeId.astype(int)
    data_df.AwayId = data_df.AwayId.astype(int)
    return data_df


def apply_season_dict(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    :param data_df:
    :return:
    """
    dict_folder = get_environment_config()["destination_folder"]
    with open(dict_folder + "season_dict.pickle", "rb") as pkl:
        season_dict = pickle.load(pkl)
    data_df[~data_df.Season.isin(season_dict)].to_csv(
        dict_folder + "reject_by_season.rej"
    )
    data_df = data_df[data_df.Season.isin(season_dict)]
    print("Фильтр сезнов осталось: ", len(data_df))
    data_df["Season"] = [season_dict[seas] for seas in data_df.Season]
    return data_df


def apply_token_filter(data_df: pd.DataFrame) -> pd.DataFrame:
    dict_folder = get_environment_config()["destination_folder"]
    with open(dict_folder + "team_league_dict.pickle", "rb") as pkl:
        team_league_dict = pickle.load(pkl)
    valid_match = [
        (str(hid) + ":" + season in team_league_dict)
        & (str(aid) + ":" + season in team_league_dict)
        for hid, aid, season in zip(data_df.HomeId, data_df.AwayId, data_df.Season)
    ]
    data_df[[not (mv) for mv in valid_match]].to_csv(
        dict_folder + "reject_by_token.rej"
    )
    data_df = data_df[valid_match]
    data_df.loc[:, ("resident_league_home")] = [
        team_league_dict[str(rlh) + ":" + seas]
        for rlh, seas in zip(data_df.HomeId, data_df.Season)
    ]
    data_df.loc[:, ("resident_league_away")] = [
        team_league_dict[str(rla) + ":" + seas]
        for rla, seas in zip(data_df.AwayId, data_df.Season)
    ]
    print("Фильтр токенизации истории, осталось: ", len(data_df))
    return data_df


def set_league_and_rest(data_df: pd.DataFrame) -> pd.DataFrame:
    dict_folder = get_environment_config()["destination_folder"]
    with open(dict_folder + "team_GId_dict.pickle", "rb") as ttd:
        team_GId_dict = pickle.load(ttd)
    data_df.loc[:, ("team_rest_home")] = [
        team_GId_dict[team][team_GId_dict[team]["last_index"]][1]
        for team in data_df.HomeId
    ]
    data_df.loc[:, ("team_rest_away")] = [
        team_GId_dict[team][team_GId_dict[team]["last_index"]][1]
        for team in data_df.AwayId
    ]
    # Отдых команд разделяем на 3 группы
    data_df.loc[:, ("team_rest_home_adj")] = [
        0 if tm < 5 else 1 if tm < 13 else 2 for tm in data_df["team_rest_home"].dt.days
    ]
    data_df.loc[:, ("team_rest_away_adj")] = [
        0 if tm < 5 else 1 if tm < 13 else 2 for tm in data_df["team_rest_away"].dt.days
    ]
    print("Время между матчами определено")
    return data_df


def set_current_idx(data_df: pd.DataFrame) -> pd.DataFrame:
    dict_folder = get_environment_config()["destination_folder"]
    with open(dict_folder + "idx_home_current_dict.pickle", "rb") as pkl:
        idx_home_current_dict = pickle.load(pkl)
    with open(dict_folder + "idx_away_current_dict.pickle", "rb") as pkl:
        idx_away_current_dict = pickle.load(pkl)

    valid_idx = [
        (home + ":" + away + ":" + str(rest_home) in idx_home_current_dict)
        & (home + ":" + away + ":" + str(rest_away) in idx_away_current_dict)
        for home, away, rest_home, rest_away in zip(
            data_df.resident_league_home,
            data_df.resident_league_away,
            data_df.team_rest_home_adj,
            data_df.team_rest_away_adj,
        )
    ]
    data_df[[not (mv) for mv in valid_idx]].to_csv(
        dict_folder + "reject_by_current_idx.rej"
    )
    data_df = data_df[valid_idx]
    print("Фильтр токенезации текущего матча, осталось: ", len(data_df))
    data_df["home_idx_current"] = [
        idx_home_current_dict[home + ":" + away + ":" + str(rest_home)]
        for home, away, rest_home in zip(
            data_df.resident_league_home,
            data_df.resident_league_away,
            data_df.team_rest_home_adj,
        )
    ]
    data_df["away_idx_current"] = [
        idx_away_current_dict[home + ":" + away + ":" + str(rest_away)]
        for home, away, rest_away in zip(
            data_df.resident_league_home,
            data_df.resident_league_away,
            data_df.team_rest_away_adj,
        )
    ]
    return data_df


def idx_recursive(
    current_team: int,
    current_index: int,
    loop_back: int,
    main_dict: dict,
    final_list=None,
) -> list:
    """
    :param current_team:
    :param current_index:
    :param loop_back:
    :param main_dict:
    :param final_list:
    :return:
    """
    last_call = False
    if final_list is None:
        last_call = True
        final_list = []
    if current_index == -1:
        current_index = main_dict[current_team]["last_index"]
        previous_index = main_dict[current_team][current_index][0]
    else:
        previous_index = main_dict[current_team][current_index][0]
    if previous_index == -1:
        if last_call:
            final_list = [0] * loop_back
            return final_list
        else:
            final_list = [0] * (loop_back - 1)
            return final_list
    loop_back -= 1
    if loop_back > 1:
        final_list = idx_recursive(
            current_team,
            previous_index,
            loop_back,
            main_dict=main_dict,
            final_list=final_list,
        )
    final_list.append(main_dict[current_team][current_index][-1])
    if last_call:
        final_list.append(main_dict[current_team]["last_idx"])
    return final_list


def prepare_for_update(data_df: pd.DataFrame) -> pd.DataFrame:
    print("Обрабатываю результат матча....")
    data_df["Score1"] = data_df["Score1"].astype(int)
    data_df["Score2"] = data_df["Score2"].astype(int)
    data_df["sum_score"] = data_df["Score1"] + data_df["Score2"]
    data_df["sum_score_k"] = [
        1 if score_k < 11 else score_k / 10 for score_k in data_df["sum_score"]
    ]
    data_df["home_score_adj"] = (data_df["Score1"] / data_df["sum_score_k"]).astype(int)
    data_df["away_score_adj"] = (data_df["Score2"] / data_df["sum_score_k"]).astype(int)
    data_df["score_adj"] = (
        data_df["home_score_adj"].astype(str)
        + "-"
        + data_df["away_score_adj"].astype(str)
    )
    return data_df


def tokenize_result(data_df: pd.DataFrame) -> pd.DataFrame:
    print("Токенизирую результаты...")
    dict_folder = get_environment_config()["destination_folder"]
    with open(dict_folder + "idx_home_dict.pickle", "rb") as pkl:
        idx_home_dict = pickle.load(pkl)
    with open(dict_folder + "idx_away_dict.pickle", "rb") as pkl:
        idx_away_dict = pickle.load(pkl)
    #####################################################
    data_df["home_idx"] = [
        idx_home_dict[
            info[0] + ":" + str(info[1]) + ":" + str(info[2]) + ":" + str(info[3])
        ]
        if info[0] + ":" + str(info[1]) + ":" + str(info[2]) + ":" + str(info[3])
        in idx_home_dict
        else 0
        for info in data_df[
            [
                "score_adj",
                "resident_league_home",
                "resident_league_away",
                "team_rest_home_adj",
            ]
        ].values
    ]
    data_df["home_idx"] = data_df["home_idx"].astype(int)
    #####################################################
    data_df["away_idx"] = [
        idx_away_dict[
            info[0] + ":" + str(info[1]) + ":" + str(info[2]) + ":" + str(info[3])
        ]
        if info[0] + ":" + str(info[1]) + ":" + str(info[2]) + ":" + str(info[3])
        in idx_away_dict
        else 0
        for info in data_df[
            [
                "score_adj",
                "resident_league_home",
                "resident_league_away",
                "team_rest_away_adj",
            ]
        ].values
    ]
    data_df["away_idx"] = data_df["away_idx"].astype(int)
    #####################################################
    zero_token = (data_df["home_idx"] == 0).sum() + (data_df["away_idx"] == 0).sum()
    return data_df


def update_matches_connections(data_df: pd.DataFrame) -> dict:
    """
    :param data_df:
    :return: dict
    """
    dict_folder = get_environment_config()["destination_folder"]
    with open(dict_folder + "team_GId_dict.pickle", "rb") as ttd:
        team_GId_dict = pickle.load(ttd)
    zero_diff = 0
    for info in tqdm(
        zip(
            data_df.date,
            data_df.HomeId,
            data_df.AwayId,
            data_df.Id,
            data_df.home_idx,
            data_df.away_idx,
        ),
        total=len(data_df),
    ):
        # 1. Проверить если ID команды в словаре, если нет перейти к добавлению
        if info[1] in team_GId_dict:
            # 2. Проверить время, если время позднее последнего добаления,
            # то можно просто присоединить снизу, инфо о последнем матче команды,
            # иначе перейти во вставку матча между матчами
            if info[3] not in team_GId_dict[info[1]]:
                if info[0] >= team_GId_dict[info[1]]["last_time"]:
                    # 3. Добавление матча для команды в словарь
                    previous_num = team_GId_dict[info[1]]["last_index"]
                    previous_time = team_GId_dict[info[1]]["last_time"]
                    previous_home = team_GId_dict[info[1]]["last_home"]
                    previous_idx = team_GId_dict[info[1]]["last_idx"]
                    # 4. Обновление блока последнего матча для команды
                    team_GId_dict[info[1]]["last_index"] = info[3]
                    team_GId_dict[info[1]]["last_time"] = info[0]
                    team_GId_dict[info[1]]["last_home"] = 0  # 0 for home
                    team_GId_dict[info[1]]["last_idx"] = info[4]
                    # 3. Добавление матча для команды в словарь
                    team_GId_dict[info[1]].update(
                        {
                            info[3]: [
                                previous_num,
                                info[0] - previous_time,
                                previous_home,
                                previous_time,
                                previous_idx,
                            ]
                        }
                    )
                else:
                    zero_diff += 1
                    # 5. Поиск точки вхождение для матча, который оказался не новым
                    current_index = team_GId_dict[info[1]]["last_index"]
                    previous_match_time = team_GId_dict[info[1]][current_index][3]
                    previous_index = team_GId_dict[info[1]][current_index][0]
                    while (info[0] < previous_match_time) & (previous_index != -1):
                        current_index = previous_index
                        previous_index = team_GId_dict[info[1]][previous_index][0]
                        previous_match_time = team_GId_dict[info[1]][current_index][3]
                    # 6. Вставка матча и обновление соседних 2 матчей
                    team_GId_dict[info[1]].update(
                        {info[3]: team_GId_dict[info[1]][current_index]}
                    )
                    upd_GId_delta = team_GId_dict[info[1]][info[3]][1]
                    team_GId_dict[info[1]][info[3]][1] = (
                        info[0] - team_GId_dict[info[1]][info[3]][3]
                    )
                    upd_GId_delta = upd_GId_delta - team_GId_dict[info[1]][info[3]][1]
                    if upd_GId_delta == -1:
                        print("Ошибка #1 обновления словаря team_GId_dict")
                    team_GId_dict[info[1]].update(
                        {current_index: [info[3], upd_GId_delta, 0, info[0], info[4]]}
                    )

        else:
            # 3. Добавление матча для команды в словарь. Новая команда
            team_GId_dict[info[1]] = {"last_index": info[3]}
            team_GId_dict[info[1]].update({"last_time": info[0]})
            team_GId_dict[info[1]].update({"last_home": 0})  # 0 for home
            team_GId_dict[info[1]].update({"last_idx": None})
            team_GId_dict[info[1]].update(
                {
                    info[3]: [
                        -1,
                        pd.Timedelta(pd.offsets.Day(7)),
                        -1,
                        info[0] - pd.DateOffset(7),
                        0,
                    ]
                }
            )

        #############################################################################
        #############################################################################

        # 1. Проверить если ID команды в словаре, если нет перейти к добавлению
        if info[2] in team_GId_dict:
            if info[3] not in team_GId_dict[info[2]]:
                # 2. Проверить время, если время позднее последнего добаления,
                # то можно просто присоединить снизу, инфо о последнем матче команды,
                # иначе перейти во вставку матча между матчами
                if info[0] >= team_GId_dict[info[2]]["last_time"]:
                    # 3. Добавление матча для команды в словарь
                    previous_num = team_GId_dict[info[2]]["last_index"]
                    previous_time = team_GId_dict[info[2]]["last_time"]
                    previous_home = team_GId_dict[info[2]]["last_home"]
                    previous_idx = team_GId_dict[info[2]]["last_idx"]
                    # 4. Обновление блока последнего матча для команды
                    team_GId_dict[info[2]]["last_index"] = info[3]
                    team_GId_dict[info[2]]["last_time"] = info[0]
                    team_GId_dict[info[2]]["last_home"] = 1  # 0 for home
                    team_GId_dict[info[2]]["last_idx"] = info[5]
                    # 3. Добавление матча для команды в словарь
                    team_GId_dict[info[2]].update(
                        {
                            info[3]: [
                                previous_num,
                                info[0] - previous_time,
                                previous_home,
                                previous_time,
                                previous_idx,
                            ]
                        }
                    )
                else:
                    zero_diff += 1
                    # 5. Поиск точки вхождение для матча, который оказался не новым
                    current_index = team_GId_dict[info[2]]["last_index"]
                    # 5. Поиск точки вхождение для матча, который оказался не новым
                    previous_match_time = team_GId_dict[info[2]][current_index][3]
                    previous_index = team_GId_dict[info[2]][current_index][0]
                    while (info[0] < previous_match_time) & (previous_index != -1):
                        current_index = previous_index  # 5. Поиск точки вхождение для матча, который оказался не новым
                        previous_index = team_GId_dict[info[2]][previous_index][0]
                        previous_match_time = team_GId_dict[info[2]][current_index][3]
                    # 6. Вставка матча и обновление соседних 2 матчей
                    team_GId_dict[info[2]].update(
                        {info[3]: team_GId_dict[info[2]][current_index]}
                    )
                    upd_GId_delta = team_GId_dict[info[2]][info[3]][1]
                    team_GId_dict[info[2]][info[3]][1] = (
                        info[0] - team_GId_dict[info[2]][info[3]][3]
                    )
                    upd_GId_delta = upd_GId_delta - team_GId_dict[info[2]][info[3]][1]
                    if upd_GId_delta == -1:
                        print("Ошибка #1 обновления словаря team_GId_dict")
                    team_GId_dict[info[2]].update(
                        {current_index: [info[3], upd_GId_delta, 0, info[0], info[5]]}
                    )

        else:
            # 3. Добавление матча для команды в словарь. Новая команда
            team_GId_dict[info[2]] = {"last_index": info[3]}
            team_GId_dict[info[2]].update({"last_time": info[0]})
            team_GId_dict[info[2]].update({"last_home": 1})  # 0 for home
            team_GId_dict[info[2]].update({"last_idx": None})
            team_GId_dict[info[2]].update(
                {
                    info[3]: [
                        -1,
                        pd.Timedelta(pd.offsets.Day(7)),
                        -1,
                        info[0] - pd.DateOffset(7),
                        0,
                    ]
                }
            )

    print("\n")
    print("zero diff = ", zero_diff)
    return team_GId_dict
