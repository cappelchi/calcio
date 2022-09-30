from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors
from utils import get_environment_config
from utils import load_dataframe
from utils import apply_token_filter
from utils import apply_season_dict
from utils import set_league_and_rest
from utils import set_current_idx
from utils import idx_recursive
import numpy as np
import pickle






def predict(new_csv: str, start_date:str, end_date:str):
    '''
    :param new_csv:
    :param start_date:
    :param end_date:
    '''
    main_folder = get_environment_config()["destination_folder"]
    data_df = load_dataframe(main_folder  + new_csv, start_date = start_date, end_date = end_date)
    print("Загрузилось матчей: ", len(data_df))
    data_df = apply_season_dict(data_df)
    data_df = apply_token_filter(data_df)
    data_df = set_league_and_rest(data_df)
    data_df = set_current_idx(data_df)
    ##################################
    with open(main_folder + "team_GId_dict.pickle", "rb") as ttd:
        team_GId_dict = pickle.load(ttd)
    ##################################
    look_back = 5
    input_list = []
    for idx in data_df['HomeId']:
        input_list.append(idx_recursive(idx, -1, look_back, main_dict = team_GId_dict)[::-1])
    data_df[[f'home_input_{num}' for num in range(1, 1 + look_back)]] = input_list
    ##################################
    look_back = 5
    input_list = []
    for idx in data_df['AwayId']:
        input_list.append(idx_recursive(idx, -1, look_back, main_dict = team_GId_dict)[::-1])
    data_df[[f'away_input_{num}' for num in range(1, 1 + look_back)]] = input_list
    ##################################
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
    ##################################
    word_vectors = get_environment_config()['word2vec'][1].replace('.vectors.npy', '')
    print('Загружаю вектора w2v: ', word_vectors)
    wv = KeyedVectors.load(word_vectors, mmap='r')
    idx_arr = np.zeros(max(wv.key_to_index) + 1)
    for key, value in wv.key_to_index.items():
        idx_arr[key] = value + 1
    ##################################
    X_input = idx_arr[data_df[names].values].astype(np.int64)
    model_path = get_environment_config()['tf_model'][0]
    model = load_model(model_path)
    data_df['predict'] = model.predict(X_input)
    data_df.to_csv(main_folder  + new_csv + 'predict.out')
    print('Предикт посчитан и загружен в ' + main_folder  + new_csv + 'predict.out')
    ##################################

#if __name__ == "__main__":
#    pd.options.display.max_columns = 50
#    folder = 'new_csv/'
#    start_date = '20220701'
#    end_date = '20220731'
#    predict(folder, start_date, end_date)