from utils import get_environment_config
from utils import create_environment_config
from utils import load_model

def update_model(model_type: str, model_num: str):
    config_dict = get_environment_config()
    if "destination_folder" in config_dict:
        create_environment_config(
            {
                "tf_model": load_model(
                    folder_name=config_dict["destination_folder"],
                    model_num=model_num,
                    model_type=model_type,
                ),
                "model_type": model_type,
                "model_no": model_num,
            }
        )
    else:
        print('Основная директория отсутствует в конфиге')


if __name__ == "__main__":
    update_model(model_type = 'HOME', model_num = '9')