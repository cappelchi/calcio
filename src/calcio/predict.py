from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors
from utils import get_environment_config
from utils import load_dicts
import pandas as pd
import functools
from glob import glob


def idx_recursive(
    current_team: int,
    current_index: int,
    loop_back: int,
    main_dict:dict,
    final_list=None,
) -> list:
    last_call = False
    if final_list is None:
        last_call = True
        final_list = []
    if current_index == -1:
        current_index = main_dict[current_team]["last_index"]
        previous_index = main_dict[current_team][current_index][0]
    else:
        previous_index = main_dict[current_team][current_index][0]
    if previous_index < 0:
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
    final_list.append(main_dict[current_team][current_index][3])
    if last_call:
        final_list.append(main_dict[current_team]["last_idx"])
    return final_list


def predict(new_csv: str):

    dicts_list = [
        "team_time_dict",
        "team_league_dict",
        "season_dict",
        "idx_home_current_dict",
        "idx_away_current_dict",
        "idx_home_dict",
        "idx_away_dict",
    ]
    main_folder = get_environment_config()["destination_folder"]
    print(main_folder)
    (
        team_time_dict,
        team_league_dict,
        season_dict,
        idx_home_current_dict,
        idx_away_current_dict,
        idx_home_dict,
        idx_away_dict,
    ) = load_dicts(main_folder)
    data_df = pd.concat(
        map(
            functools.partial(pd.read_csv, sep=";", compression=None),
            glob(new_csv + "*.csv"),
        ),
        ignore_index=True,
    )
    print("Загрузилось матчей: ", len(data_df))
    # Оставляем только дату
    data_df[["date", "times_ext"]] = data_df.Time.str.split(expand=True)
    data_df = data_df.drop(["Time", "times_ext"], axis="columns")
    data_df.HomeId = data_df.HomeId.astype(int)
    data_df.AwayId = data_df.AwayId.astype(int)
    data_df[~data_df.Season.isin(season_dict)].to_csv(new_csv + "reject_by_season.rej")
    data_df = data_df[data_df.Season.isin(season_dict)]
    print("Фильтр сезнов: ", len(data_df))
    data_df.loc[:, "Season"] = [season_dict[seas] for seas in data_df.Season]
    valid_match = [
        (str(hid) + ":" + season in team_league_dict)
        & (str(aid) + ":" + season in team_league_dict)
        for hid, aid, season in zip(data_df.HomeId, data_df.AwayId, data_df.Season)
    ]
    data_df[[not (mv) for mv in valid_match]].to_csv(
        new_csv + "reject_by_match_token.rej"
    )
    data_df = data_df[valid_match]
    print("Фильтр валидности матчей: ", len(data_df))
    data_df.loc[:, "resident_league_home"] = [
        team_league_dict[str(rlh) + ":" + seas]
        for rlh, seas in zip(data_df.HomeId, data_df.Season)
    ]
    data_df.loc[:, "resident_league_away"] = [
        team_league_dict[str(rla) + ":" + seas]
        for rla, seas in zip(data_df.AwayId, data_df.Season)
    ]
    data_df.loc[:, "team_rest_home"] = [
        team_time_dict[team][team_time_dict[team]["last_index"]][1]
        for team in data_df.HomeId
    ]
    data_df.loc[:, "team_rest_away"] = [
        team_time_dict[team][team_time_dict[team]["last_index"]][1]
        for team in data_df.AwayId
    ]
    # Отдых команд разделяем на 3 группы
    data_df.loc[:, "team_rest_home_adj"] = [
        0 if tm < 5 else 1 if tm < 13 else 2 for tm in data_df["team_rest_home"].dt.days
    ]
    data_df.loc[:, "team_rest_away_adj"] = [
        0 if tm < 5 else 1 if tm < 13 else 2 for tm in data_df["team_rest_away"].dt.days
    ]
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
        new_csv + "reject_by_current_match_token.rej"
    )
    data_df = data_df[valid_idx]
    print("Фильтр valid_idx: ", len(data_df))
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
    ##################################
    look_back = 5
    input_list = []
    for idx in data_df['HomeId']:
        input_list.append(idx_recursive(idx, -1, look_back, main_dict = team_time_dict)[::-1])
    data_df[[f'home_input_{num}' for num in range(1, 1 + look_back)]] = input_list
    ##################################
    look_back = 5
    input_list = []
    for idx in data_df['AwayId']:
        input_list.append(idx_recursive(idx, -1, look_back, main_dict = team_time_dict)[::-1])
    data_df[[f'away_input_{num}' for num in range(1, 1 + look_back)]] = input_list
    names = ['home_idx_current',
             'home_input_1',
             'home_input_2',
             'home_input_3',
             'home_input_4',
             'home_input_5',
             'away_idx_current',
             'away_input_1',
             'away_input_2',
             'away_input_3',
             'away_input_4',
             'away_input_5']

    print("Осталось на предикт матчей: ", len(data_df))
    word_vectors = main_folder + get_environment_config()['word2vec'][1]
    wv = KeyedVectors.load(word_vectors, mmap='r')
    idx_arr = np.zeros(max(wv.key_to_index) + 1)
    for key, value in wv.key_to_index.items():
        idx_arr[key] = value + 1
    X_input = idx_arr[data_df[names].values].astype(int)
    model_path = main_folder + get_environment_config()['tf_model'][0]
    model = load_model(model_path)
    data_df['predict'] = model.predict(X_input)
    data_df.to_csv(new_csv + 'predict.out')

if __name__ == "__main__":
    predict("../../new_csv/")
