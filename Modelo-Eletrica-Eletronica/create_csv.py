import time
import multiprocessing
import concurrent.futures
import create_txt
from lxml import html
import requests
from bs4 import BeautifulSoup
import csv

eletronica_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=18&curso=18045&hab=0&pagina="

eletrica_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=18&curso=18050&hab=0&pagina="


def get_list_eletronica():
    global list_text_eletronica
    list_text_eletronica = create_txt.html_to_text_list('url_eletronica.txt')

def get_list_eletrica():
    global list_text_eletrica
    list_text_eletrica = create_txt.html_to_text_list('url_eletrica.txt')    

#---------------------------------------------------------------------------------------------------------------------------------

def main():
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(create_txt.make_txt,'url_eletronica.txt','w',eletronica_lib,42)
        executor.submit(create_txt.make_txt,'url_eletrica.txt','w',eletrica_lib,42)
    finish = time.perf_counter()
    print(f'\nURL TXT criado em: {round(finish-start, 2)} segundos')    

    start = time.perf_counter()    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(get_list_eletronica())
        executor.submit(get_list_eletrica())
    with open('train.csv','w') as tr:
        csv_writer = csv.writer(tr)
        csv_writer.writerow(['valor','texto'])     
    create_txt.make_csv('train.csv',list_text_eletronica,1)
    create_txt.make_csv('train.csv',list_text_eletrica,0)
    finish = time.perf_counter()
    print(f'\nCSV criado em: {round(finish-start, 2)} segundos\n')

if __name__ == '__main__':
    main()

#---------------------------------------------------------------------------------------------------------------------------------

