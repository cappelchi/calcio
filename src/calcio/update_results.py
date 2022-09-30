from utils import load_dataframe
from utils import get_environment_config
from utils import apply_token_filter
from utils import apply_season_dict
from utils import set_league_and_rest
from utils import prepare_for_update
from utils import tokenize_result
from utils import update_matches_connections
import pickle

def update_results(folder:str, start_date:str, end_date:str):
    '''
    :param folder:
    :param start_date:
    :param end_date:
    '''

    main_folder = get_environment_config()["destination_folder"]
    data_df = load_dataframe(main_folder  + folder, start_date = start_date, end_date = end_date)
    print("Загрузилось матчей: ", len(data_df))
    data_df = apply_season_dict(data_df)
    data_df = apply_token_filter(data_df)
    data_df = set_league_and_rest(data_df)
    data_df = prepare_for_update(data_df)
    data_df = tokenize_result(data_df)
    updated_dict = update_matches_connections(data_df)
    with open(main_folder + 'tmp.pickle', 'wb') as pkl:
        pickle.dump(updated_dict, pkl, protocol=pickle.HIGHEST_PROTOCOL)


#if __name__ == "__main__":
#    folder = 'new_csv/'
#    start_date = '20220701'
#    end_date = '20220731'
#    update_results(folder, start_date, end_date)
