import requests
from bs4 import BeautifulSoup
import csv

list_url_eletronica = ["http://www.tcc.sc.usp.br/tce/disponiveis/18/180450/tce-29082014-145751/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180450/tce-09042010-105658/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180450/tce-21022018-170526/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180450/tce-28032012-115605/?&lang=br"]



list_url_eletrica = ["http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-14032016-175537/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-06082018-161933/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-17012018-115853/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-05052010-094527/?&lang=br"]

list_text_eletronica = []
list_text_eletrica = []

def html_to_txt_list(list_url):
    list_text = []
    for element in list_url:        
        list_text.append((BeautifulSoup(requests.get(element).content,'lxml').find(id='DocumentoTextoResumo')).text)
        print (list_text)
    return list_text    

list_text_eletronica = html_to_txt_list(list_url_eletronica)
list_text_eletrica = html_to_txt_list(list_url_eletrica)

print(len(list_url_eletronica))
print(len(list_text_eletrica))

with open("eval.csv",'w',encoding="utf-8") as tr:
    csv_writer = csv.writer(tr)
    csv_writer.writerow(['valor','texto'])
    for element in list_text_eletronica:
        csv_writer.writerow([1,element])
    for element in list_text_eletrica:
        csv_writer.writerow([0,element])  