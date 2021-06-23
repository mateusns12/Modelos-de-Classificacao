from lxml import html
import requests
from bs4 import BeautifulSoup
import csv

def make_page_list(lib,num):
    list = []
    for i in range(1,num):        
        list.append(lib + str(i))
    return list

def make_url_list(url):
    list = []
    for link in url:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'lxml')
        list.append(soup)    
    return list

def get_href(url_list):
    list = []
    for element in url_list:
        for link in element.find_all('a'):
            if 'tce' in link.get('href'):
                list.append(link.get('href'))    
    return list

def make_txt(nome_txt, modo ,url_lib,num):
    with open(nome_txt,modo,encoding='utf-8') as url:
        for element in get_href(make_url_list(make_page_list(url_lib,num))):
            url.write(element)
            url.write('\n')
    return True

#-----------------------------------------------------------------------------------------------------------

def read_txt(nome_txt):
    list = []
    with open(nome_txt,'r') as file:
        for line in file:
            list.append(line)
    return list        

def html_to_text_list(nome_txt):
    list_text = []
    for element in read_txt(nome_txt): 
        try:       
            list_text.append((BeautifulSoup(requests.get(element).content,'lxml').find(id='DocumentoTextoResumo')).text)
        except:
            pass    
    return list_text    

#--------------------------------------------------------------------------------------------------------------------------------
def make_csv(nome_csv,list,modo):
    with open(nome_csv,"a",encoding="utf-8") as tr:
        csv_writer = csv.writer(tr)        
        for element in list:
            try:
                csv_writer.writerow([modo,str(element)])
            except:
                pass    
