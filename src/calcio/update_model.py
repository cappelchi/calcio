from utils import get_credential
from utils import get_environment_config


def update_model(model_type:str, model_num:str):
    _, api_key = get_credential()
    neptune_model = f"FOOT-" + model_type
    neptune_model_version = f"FOOT-" + model_type + "-" + model_num
    model_version = neptune.init_model_version(
        project="scomesse/football",
        model=neptune_model,
        api_token=api_key,
        version=neptune_model_version,
    )
    print(f"Загружаем модель {model_type} n.{model_num}")
    model_version["model"].download(path_to_model)
    model_version.stop()
    logging.info(f"Downloaded update: {path_to_model}")