from CSV2TXT import csv2txt
import pandas as pd

save_path = './txt_file/txt_data.txt'

data = pd.read_csv('./processed_file(1) (9).csv')

c2t = csv2txt(data, save_path)

c2t.transform()