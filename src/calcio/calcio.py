from set_environment import set_environment
from update_model import update_model
from update_word2vec import update_word2vec
from predict import predict
from update_results import update_results
import click

@click.command()
@click.option('--set-environment', 'command', flag_value = 'set_environment')
@click.option('--change-model', 'command', flag_value = 'change_model')
@click.option('--load-w2v', 'command', flag_value = 'load_w2v')
@click.option('--predict', 'command', flag_value = 'predict')
@click.option('--update-results', 'command', flag_value = 'update_results')
@click.option('--folder', '-f', default = './', help = 'Define main folder')
@click.option('--model-type', '-m', default = 'HOME', help = 'Define model type HOME, DRAW, AWAY')
@click.option('--version', '-v', default = '1', help = 'Define model version 1, 2, 3, ...')
@click.option('--w2v-name', '-w', default = 'word2vec_220811', help = 'Define model version 1, 2, 3, ...')
@click.option('--start-date', '-w', default = '', help = 'Define start date of predict or update')
@click.option('--end-date', '-w', default = '', help = 'Define end date of predict or update')
def main(**params):
    if params['command'] == 'set_environment':
        set_environment(destination_folder = params['folder'])
    elif params['command'] == 'change_model':
        update_model(params['model_type'], params['version'])
    elif params['command'] == 'load_w2v':
        update_word2vec(saved_name = params['w2v_name'])
    elif params['command'] == 'predict':
        predict(params['folder'], params['start_date'], params['end_date'])
    elif params['command'] == 'update_results':
        update_results(params['folder'], params['start_date'], params['end_date'])

if __name__ == "__main__":
    main()