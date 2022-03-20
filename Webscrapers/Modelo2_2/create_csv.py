import time
import multiprocessing
import concurrent.futures
import create_txt
from lxml import html
import requests
from bs4 import BeautifulSoup
import csv

direito_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=89&curso=89001&hab=0&pagina="

eletrica_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=18&curso=18050&hab=0&pagina="

def get_list_el():
    global list_text_el
    list_text_el = create_txt.html_to_text_list('url_el.txt')

def get_list_dir():
    global list_text_dir
    list_text_dir = create_txt.html_to_text_list('url_dir.txt')    

#---------------------------------------------------------------------------------------------------------------------------------

def main():
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(create_txt.make_txt,'url_dir.txt','w',direito_lib,21)
        executor.submit(create_txt.make_txt,'url_el.txt','w',eletrica_lib,21)
    finish = time.perf_counter()
    print(f'\nURL TXT criado em: {round(finish-start, 2)} segundos')    

    start = time.perf_counter()    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(get_list_el())
        executor.submit(get_list_dir())
    with open('train.csv','w') as tr:
        csv_writer = csv.writer(tr)
        csv_writer.writerow(['valor','texto'])     
    create_txt.make_csv('train.csv',list_text_el,1)
    create_txt.make_csv('train.csv',list_text_dir,0)
    finish = time.perf_counter()
    print(f'\nCSV criado em: {round(finish-start, 2)} segundos\n')

if __name__ == '__main__':
    main()

#---------------------------------------------------------------------------------------------------------------------------------

