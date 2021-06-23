import requests
from bs4 import BeautifulSoup
import csv

list_url_el = ["http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-17112011-093400/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-20082010-113011/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-19032019-111327/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-01022016-152430/?&lang=br"]

"""
list_url_el = ["http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-21082018-171725/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-22012014-103159/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-29032012-095846/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-10012017-165059/?&lang=br"]
"""

list_url_dir = ["http://www.tcc.sc.usp.br/tce/disponiveis/89/890010/tce-22052017-164502/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/89/890010/tce-29042015-144528/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/89/890010/tce-16122013-105154/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/89/890010/tce-11052015-201819/?&lang=br"]

list_text_el = []
list_text_dir = []

def html_to_txt_list(list_url):
    list_text = []
    for element in list_url:        
        list_text.append((BeautifulSoup(requests.get(element).content,'lxml').find(id='DocumentoTextoResumo')).text)
    return list_text    

list_text_el = html_to_txt_list(list_url_el)
list_text_dir = html_to_txt_list(list_url_dir)

print(len(list_text_el))
print(len(list_text_dir))

with open("eval.csv",'w',encoding="utf-8") as tr:
    csv_writer = csv.writer(tr)
    csv_writer.writerow(['valor','texto'])
    for element in list_text_dir:
        csv_writer.writerow([0,element])
    for element in list_text_el:
        csv_writer.writerow([1,element])  