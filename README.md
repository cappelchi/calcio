# calcio
необходимо в корневу папку положить файл: credential.txt
1. развернуть окружение по умолчанию в папку ./: <br>
calcio --set_environment
calcio --set_environment=./calcio_folder

2. выбрать модель для предикта (по умолчанию HOME версия 2):
    - тип:  HOME, DRAW, AWAY
    - версия: 1, 2, ...<br>
calcio --update_model --type=HOME --version=10

3. поменять эмбеддинги word2vec:
    - name: word2vec_220811<br>
calcio --update_embedding=word2vec_220811

4. запустить предикт:
    - folder: /path/to/folder
    - file: /path/to/file.csv<br>
calcio --predict --folder=/path/to/folder
calcio --predict --file=/path/to/file.csv

5. запустить обновление результатов
    - folder: /path/to/folder
    - file: /path/to/file.csv<br>
calcio --update-results --folder=/path/to/folder
calcio --update-results --file=/path/to/file.csv
