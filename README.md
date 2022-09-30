# calcio
необходимо в корневую папку положить файл: credential.txt
1. развернуть окружение по умолчанию в папку ./: <br>

calcio --set_environment<br>
calcio --set_environment --folder ./calcio_folder<br>
calcio --set_environment -f ./calcio_folder

2. выбрать модель для предикта (по умолчанию HOME версия 2):
    - тип:  HOME, DRAW, AWAY
    - версия: 1, 2, ...<br>

calcio --update_model --model_type AWAY --version 10
calcio --update_model -m AWAY -v 10

3. поменять эмбеддинги word2vec:
    - name: word2vec_220811<br>

calcio --load_w2v --w2v_name word2vec_220811
calcio --load_w2v --w word2vec_220811

4. запустить предикт:
    - folder: /path/to/folder
    - file: /path/to/file.csv<br>

calcio --predict --folder /path/to/folder --start-date --end-date<br>
calcio --predict --file /path/to/file.csv

5. запустить обновление результатов
    - folder: /path/to/folder
    - file: /path/to/file.csv<br>

calcio --update-results --folder /path/to/folder --start-date --end-date<br>
calcio --update-results --file /path/to/file.csv
