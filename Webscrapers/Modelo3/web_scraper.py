import time
import multiprocessing
import concurrent.futures
import create_txt
from lxml import html
import requests
from bs4 import BeautifulSoup
import csv

direito_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=89&curso=89001&hab=0&pagina="
#20

eletrica_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=18&curso=18050&hab=0&pagina="
#44

prod_mecanica_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=18&curso=18083&hab=0&pagina="
#24

odontologia_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=58&curso=58012&hab=0&pagina="
#11

eng_ambiental_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=18&curso=18030&hab=0&pagina="
#36

geografia_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=8&curso=8021&hab=104&pagina="
#20

eng_computacao_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=97&curso=97001&hab=0&pagina="
#13

eletronica_lib = "http://www.tcc.sc.usp.br/index.php?option=com_jumi&fileid=17&Itemid=178&lang=br&id=18&curso=18045&hab=0&pagina="
#45

list_lib = [eletronica_lib,
            direito_lib,
            eletrica_lib,
            odontologia_lib,
            eng_computacao_lib,
            geografia_lib,
            eng_ambiental_lib,
            prod_mecanica_lib]

list_labels = [1,2,3,4,5,6,7,8]

#list_pages = [45,20,44,11,13,20,36,24]

list_pages = [20,20,20,20,20,20,20,20]

list_nome_arquivo = ["url_eletronica.txt",
                    "url_direito.txt",
                    "url_eletrica.txt",
                    "url_odontologia.txt",
                    "url_computacao.txt",
                    "url_geografia.txt",
                    "url_ambiental.txt",
                    "url_mecanica.txt"]

zip_to_dict_lib = [list(i) for i in zip(list_lib,list_labels,list_nome_arquivo)]

dict_lib = {"eletronica":zip_to_dict_lib[0],
            "direito":zip_to_dict_lib[1],
            "eletrica":zip_to_dict_lib[2],
            "odontologia":zip_to_dict_lib[3],
            "computacao":zip_to_dict_lib[4],
            "geografia":zip_to_dict_lib[5],
            "ambiental":zip_to_dict_lib[6],
            "mecanica":zip_to_dict_lib[7]}

list_text_eletronica = []
list_text_direito = []
list_text_eletrica = []
list_text_odontologia = []
list_text_computacao = []
list_text_geografia = []
list_text_ambiental = []
list_text_mecanica = []

#---------------------------------------------------------------------------------------------------------------------------------

def main():
    
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in zip(list_nome_arquivo,list_lib,list_pages):
            executor.submit(create_txt.make_txt,i[0],'w',i[1],i[2]+1)

    finish = time.perf_counter()
    print(f'\nURL TXT criado em: {round(finish-start, 2)} segundos')    
    
    

    start = time.perf_counter()    
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    
    list_text_eletronica = create_txt.html_to_text_list(list_nome_arquivo[0])
    list_text_direito = create_txt.html_to_text_list(list_nome_arquivo[1])
    list_text_eletrica = create_txt.html_to_text_list(list_nome_arquivo[2])
    list_text_odontologia = create_txt.html_to_text_list(list_nome_arquivo[3])
    list_text_computacao = create_txt.html_to_text_list(list_nome_arquivo[4])
    list_text_geografia = create_txt.html_to_text_list(list_nome_arquivo[5])
    list_text_ambiental = create_txt.html_to_text_list(list_nome_arquivo[6])
    list_text_mecanica = create_txt.html_to_text_list(list_nome_arquivo[7])

    list_of_lists = [list_text_eletronica,list_text_direito,list_text_eletrica,list_text_odontologia,
                list_text_computacao,list_text_geografia,list_text_ambiental,list_text_mecanica]         

    with open('train.csv','w') as tr:
        csv_writer = csv.writer(tr)
        csv_writer.writerow(['valor','texto'])    

    for element in zip(list_of_lists,list_labels):
        create_txt.make_csv('train.csv',element[0],element[1])
    
        #create_txt.make_csv('train.csv',list_text_dir,0)
    finish = time.perf_counter()
    print(f'\nCSV criado em: {round(finish-start, 2)} segundos\n')

if __name__ == '__main__':
    main()