from utils import load_dataframe
from utils import get_environment_config
from utils import set_league_and_rest
from utils import prepare_for_update
from utils import tokenize_result
from utils import update_matches_connections
from utils import update_dict_with_new_matches_tokens
import pickle

def update_results(folder:str, start_date:str, end_date:str):
    '''
    :param folder:
    :param start_date:
    :param end_date:
    '''

    main_folder = get_environment_config()["destination_folder"]
    data_df = load_dataframe(folder, start_date = start_date, end_date = end_date)
    print("Загрузилось матчей: ", len(data_df))
    updated_dict = update_matches_connections(data_df)
    data_df = set_league_and_rest(data_df, updated_dict)
    data_df = prepare_for_update(data_df)
    data_df = tokenize_result(data_df)
    updated_dict = update_dict_with_new_matches_tokens(data_df, updated_dict)

    print('Сохраняется словарь истории матчей...')
    with open(main_folder + 'team_GId_dict.pickle', 'wb') as pkl:
        pickle.dump(updated_dict, pkl, protocol=pickle.HIGHEST_PROTOCOL)

