# calcio
1. развернуть окружение по умолчанию в папку ./: <br>

calcio --set-environment<br>
calcio --set-environment --folder ./calcio_folder<br>
calcio --set-environment -f ./calcio_folder

2. необходимо в папку /calcio/ рядом с README положить файл: credential.txt
2. выбрать модель для предикта (по умолчанию HOME версия 1):
    - тип:  HOME, DRAW, AWAY
    - версия: 1, 2, ...<br>

calcio --change-model --model_type AWAY --version 10
calcio --change-model -m AWAY -v 10

3. поменять эмбеддинги word2vec:
    - name: word2vec_220811<br>

calcio --load-w2v --w2v_name word2vec_220811
calcio --load-w2v --w word2vec_220811

4. запустить предикт:
    - folder: /path/to/folder
    - start-date: yyyymmdd
    - end-date: yyyymmdd (опционально)

calcio --predict --folder /path/to/folder --start-date --end-date<br>
calcio --predict --folder /path/to/folder --start-date (для 1 дня<br>

5. запустить обновление результатов
    - folder: /path/to/folder
    - start-date: yyyymmdd
    - end-date: yyyymmdd (опционально)

calcio --update-results --folder /path/to/folder --start-date --end-date<br>
calcio --update-results --file /path/to/file.csv
