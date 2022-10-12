import os
import tarfile
import logging
import pickle
import functools
import pandas as pd
import yaml
import neptune.new as neptune

pd.options.mode.chained_assignment = None
CONFIG_PATH = './calcio/config.yaml'


def get_credential(frmwork="neptune_team"):
    token_path = os.path.realpath("./calcio/credential.txt")
    with open(token_path, "r") as container:
        for line in container:
            if frmwork in line:
                login, psw = line.split(" ")[1], line.split(" ")[2].split("\n")[0]
                return login, psw


def get_environment_config() -> dict:
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as conf:
            current_config = yaml.load(conf, Loader=yaml.SafeLoader)
        return current_config
    else:
        print("Нет конфига, запустите развёртывание окружения")
        return {}


def set_config(config_dict: dict) -> dict:
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as conf:
            current_config = yaml.load(conf, Loader=yaml.SafeLoader)
        for key, value in config_dict.items():
            current_config[key] = value
        with open(CONFIG_PATH, "w") as conf:
            yaml.dump(current_config, conf)
    else:
        print("Нет конфига, запустите развёртывание окружения")


def create_environment_config(config_dict: dict):
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as conf:
            current_config = yaml.load(conf, Loader=yaml.SafeLoader)
        for key, value in config_dict.items():
            current_config[key] = value
        with open(CONFIG_PATH, "w") as conf:
            yaml.dump(current_config, conf)
    else:
        with open(CONFIG_PATH, "w") as conf:
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
        print(f"Файл {file_path} не имеет расширения tar.gz")
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
    if end_date == "":
        end_date = start_date
    data_csv_list = [
        folder + str(dd).replace("-", "") + ".csv"
        for dd in pd.date_range(start=start_date, end=end_date).date
    ]
    data_df = pd.concat(
        map(functools.partial(pd.read_csv, sep=";", compression=None), data_csv_list),
        ignore_index=True,
    )
    # Обновление
    data_df["timestamp"] = (
        pd.to_datetime(data_df["BeginTime"], dayfirst=True).astype("int64") // 10**9
    )
    data_df[["date", "times_ext"]] = data_df["BeginTime"].str.split(expand=True)
    data_df = data_df.drop(["BeginTime", "times_ext"], axis="columns")
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
    data_df["Season"] = data_df["Season"].astype(str)
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


def set_league_and_rest(data_df: pd.DataFrame, team_GId_dict: dict) -> pd.DataFrame:
    """

    :param data_df:
    :param team_GId_dict:
    :return:
    """
    timestamp7days = 7 * 24 * 60 * 60
    data_df["team_rest_home"] = [
        team_GId_dict[team][idx][1]
        - team_GId_dict[team][team_GId_dict[team][idx][0]][1]
        if team_GId_dict[team][idx][0] != -1
        else timestamp7days
        for team, idx in zip(data_df.HomeId, data_df.Id)
    ]
    data_df["team_rest_away"] = [
        team_GId_dict[team][idx][1]
        - team_GId_dict[team][team_GId_dict[team][idx][0]][1]
        if team_GId_dict[team][idx][0] != -1
        else timestamp7days
        for team, idx in zip(data_df.AwayId, data_df.Id)
    ]
    # Отдых команд разделяем на 3 группы
    data_df["team_rest_home_adj"] = [
        0 if tm < 500_000 else 1 if tm < 1_000_000 else 2
        for tm in data_df["team_rest_home"]
    ]
    data_df["team_rest_away_adj"] = [
        0 if tm < 500_000 else 1 if tm < 1_000_000 else 2
        for tm in data_df["team_rest_away"]
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
    if final_list is None:
        final_list = []
    previous_index = main_dict[current_team][current_index][0]
    if previous_index == -1:
        final_list = [0] * loop_back
        return final_list
    if len(main_dict[current_team][previous_index]) == 3:
        previous_idx = main_dict[current_team][previous_index][2]
    else:
        #Возможно за указанный период команда играет более одного матча
        print('Попытка включить в цепочку матч без результата. Кодирую как 0')
        print('Команда: ', current_team, ' Матч: ', previous_index, ' Ссылка: ', current_index)
        print(len(main_dict[current_team][previous_index]))
        previous_idx = 0
    loop_back -=1
    if loop_back > 0:
        final_list = idx_recursive(current_team,
                                   previous_index,
                                   loop_back,
                                   main_dict = main_dict,
                                   final_list = final_list)
    final_list.append(previous_idx)

    return final_list


def prepare_for_update(data_df: pd.DataFrame) -> pd.DataFrame:
    print("Обрабатываю результат матча....")
    data_df["Result1"] = data_df["Result1"].astype(int)
    data_df["Result2"] = data_df["Result2"].astype(int)
    data_df["sum_score"] = data_df["Result1"] + data_df["Result2"]
    data_df["sum_score_k"] = [
        1 if score_k < 11 else score_k / 10 for score_k in data_df["sum_score"]
    ]
    data_df["home_score_adj"] = (data_df["Result1"] / data_df["sum_score_k"]).astype(
        int
    )
    data_df["away_score_adj"] = (data_df["Result2"] / data_df["sum_score_k"]).astype(
        int
    )
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
    print("Токенизировано в 0:", zero_token)
    return data_df


def update_matches_connections(data_df: pd.DataFrame) -> dict:
    """
    :param data_df:
    :return: dict
    """
    data_df = data_df.sort_values(by="timestamp").reset_index(drop=True)
    timestamp7days = 7 * 24 * 60 * 60
    dict_folder = get_environment_config()["destination_folder"]
    with open(dict_folder + "team_GId_dict.pickle", "rb") as ttd:
        team_GId_dict = pickle.load(ttd)
    for info in zip(
        data_df.timestamp,
        data_df.HomeId,
        data_df.AwayId,
        data_df.Id,
    ):
        time_stamp = info[0]
        homeID = info[1]
        awayID = info[2]
        matchID = info[3]
        # 1. Проверить если ID команды в словаре, если нет перейти к добавлению
        if homeID in team_GId_dict:
            # 2. Проверить время, если время позднее последнего добаления,
            # то можно просто присоединить снизу, инфо о последнем матче команды,
            # иначе перейти во вставку матча между матчами
            if matchID not in team_GId_dict[homeID]:
                if time_stamp >= team_GId_dict[homeID]["last_time"]:
                    # 3. Добавление матча для команды в словарь
                    previous_num = team_GId_dict[homeID]["last_index"]
                    # 4. Обновление блока последнего матча для команды
                    team_GId_dict[homeID]["last_index"] = matchID
                    team_GId_dict[homeID]["last_time"] = time_stamp
                    # 3. Добавление матча для команды в словарь
                    team_GId_dict[homeID].update(
                        {
                            matchID: [
                                previous_num,
                                time_stamp,
                            ]
                        }
                    )
                else:
                    # 5. Поиск точки вхождение для матча, который оказался не новым
                    current_index = team_GId_dict[homeID]["last_index"]
                    previous_match_time = team_GId_dict[homeID][current_index][1]
                    previous_index = team_GId_dict[homeID][current_index][0]
                    while (info[0] < previous_match_time) & (previous_index != -1):
                        current_index = previous_index
                        previous_index = team_GId_dict[homeID][previous_index][0]
                        previous_match_time = team_GId_dict[homeID][current_index][1]
                    # 6. Вставка матча и обновление соседних 2 матчей
                    team_GId_dict[homeID].update(
                        {matchID: [team_GId_dict[homeID][current_index][0], time_stamp]}
                    )
                    team_GId_dict[homeID].update({current_index: [matchID, time_stamp]})

        else:
            # 3. Добавление матча для команды в словарь. Новая команда
            team_GId_dict[homeID] = {"last_index": info[3]}
            team_GId_dict[homeID].update({"last_time": time_stamp})
            team_GId_dict[homeID].update({matchID: [-1, time_stamp - timestamp7days]})

        #############################################################################
        #############################################################################

        if awayID in team_GId_dict:
            # 2. Проверить время, если время позднее последнего добаления,
            # то можно просто присоединить снизу, инфо о последнем матче команды,
            # иначе перейти во вставку матча между матчами
            if matchID not in team_GId_dict[awayID]:
                if time_stamp >= team_GId_dict[awayID]["last_time"]:
                    # 3. Добавление матча для команды в словарь
                    previous_num = team_GId_dict[awayID]["last_index"]
                    # 4. Обновление блока последнего матча для команды
                    team_GId_dict[awayID]["last_index"] = matchID
                    team_GId_dict[awayID]["last_time"] = time_stamp
                    # 3. Добавление матча для команды в словарь
                    team_GId_dict[awayID].update(
                        {
                            matchID: [
                                previous_num,
                                time_stamp,
                            ]
                        }
                    )
                else:
                    # 5. Поиск точки вхождение для матча, который оказался не новым
                    current_index = team_GId_dict[awayID]["last_index"]
                    previous_match_time = team_GId_dict[awayID][current_index][1]
                    previous_index = team_GId_dict[awayID][current_index][0]
                    while (time_stamp < previous_match_time) & (previous_index != -1):
                        current_index = previous_index
                        previous_index = team_GId_dict[awayID][previous_index][0]
                        previous_match_time = team_GId_dict[awayID][current_index][1]
                    # 6. Вставка матча и обновление соседних 2 матчей
                    team_GId_dict[awayID].update(
                        {matchID: [team_GId_dict[awayID][current_index][0], time_stamp]}
                    )
                    team_GId_dict[awayID].update({current_index: [matchID, time_stamp]})

        else:
            # 3. Добавление матча для команды в словарь. Новая команда
            team_GId_dict[awayID] = {"last_index": info[3]}
            team_GId_dict[awayID].update({"last_time": time_stamp})
            team_GId_dict[awayID].update({matchID: [-1, time_stamp - timestamp7days]})

    return team_GId_dict


def update_dict_with_new_matches_tokens(data_df: pd.DataFrame, team_GId_dict: dict):
    """
    :param data_df:
    :param team_GId_dict:
    :return:
    """
    dict_folder = get_environment_config()["destination_folder"]
    errors = 0
    errors_list = []
    for info in zip(
        data_df.HomeId, data_df.AwayId, data_df.Id, data_df.home_idx, data_df.away_idx
    ):
        homeID = info[0]
        awayID = info[1]
        matchID = info[2]
        homeIDX = info[3]
        awayIDX = info[4]
        team_GId_dict[homeID][matchID] += [homeIDX]
        if len(team_GId_dict[homeID][matchID]) > 3:
            errors += 1
            errors_list.append([homeID, matchID])
        team_GId_dict[awayID][matchID] += [awayIDX]
        if len(team_GId_dict[awayID][matchID]) > 3:
            errors += 1
            errors_list.append([awayID, matchID])

        print("Ошибок добавления индекса в словарь: ", errors)
        if errors > 0:
            pd.DataFrame(errors_list, columns=["teamID", "matchID"]).to_csv(
                dict_folder + "index_errors.csv"
            )

        return team_GId_dict
