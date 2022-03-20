import requests
from bs4 import BeautifulSoup
import csv

list_url_eletronica = ["http://www.tcc.sc.usp.br/tce/disponiveis/18/180450/tce-29082014-145751/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180450/tce-09042010-105658/?&lang=br",]

list_url_direito = ["http://www.tcc.sc.usp.br/tce/disponiveis/89/890010/tce-01062015-144731/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/89/890010/tce-26052015-195817/?&lang=br"]

list_url_eletrica = ["http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-14032016-175537/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180500/tce-06082018-161933/?&lang=br",]

list_url_odontologia = ["http://www.tcc.sc.usp.br/tce/disponiveis/58/580120/tce-06012020-115502/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/58/580120/tce-06012020-115437/?&lang=br"]

list_url_computacao = ["http://www.tcc.sc.usp.br/tce/disponiveis/97/970010/tce-29032010-101614/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/97/970010/tce-17012018-182554/?&lang=br"]

list_url_geografia = ["http://www.tcc.sc.usp.br/tce/disponiveis/8/8021101/tce-07062019-165418/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/8/8021101/tce-07062019-170802/?&lang=br"]

list_url_ambiental = ["http://www.tcc.sc.usp.br/tce/disponiveis/18/180300/tce-29042019-095947/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180300/tce-16072010-105433/?&lang=br"]

list_url_mecanica = ["http://www.tcc.sc.usp.br/tce/disponiveis/18/180830/tce-25052010-113426/?&lang=br",
"http://www.tcc.sc.usp.br/tce/disponiveis/18/180830/tce-07012015-143549/?&lang=br"]

list_text_eletronica = []
list_text_direito = []
list_text_eletrica = []
list_text_odontologia = []
list_text_computacao = []
list_text_geografia = []
list_text_ambiental = []
list_text_mecanica = []

def html_to_txt_list(list_url):
    list_text = []
    for element in list_url:        
        list_text.append((BeautifulSoup(requests.get(element).content,'lxml').find(id='DocumentoTextoResumo')).text)
    return list_text    

list_text_eletronica = html_to_txt_list(list_url_eletronica)
list_text_direito = html_to_txt_list(list_url_direito)
list_text_eletrica = html_to_txt_list(list_url_eletrica)
list_text_odontologia = html_to_txt_list(list_url_odontologia)
list_text_computacao = html_to_txt_list(list_url_computacao)
list_text_geografia = html_to_txt_list(list_url_geografia)
list_text_ambiental = html_to_txt_list(list_url_ambiental)
list_text_mecanica = html_to_txt_list(list_url_mecanica)

with open("eval.csv",'w',encoding="utf-8") as tr:
    csv_writer = csv.writer(tr)
    csv_writer.writerow(['valor','texto'])
    for element in list_text_eletronica:
        csv_writer.writerow([1,element])
    for element in list_text_direito:
        csv_writer.writerow([2,element])  
    for element in list_text_eletrica:
        csv_writer.writerow([3,element]) 
    for element in list_text_odontologia:
        csv_writer.writerow([4,element]) 
    for element in list_text_computacao:
        csv_writer.writerow([5,element]) 
    for element in list_text_geografia:
        csv_writer.writerow([6,element]) 
    for element in list_text_ambiental:
        csv_writer.writerow([7,element]) 
    for element in list_text_mecanica:
        csv_writer.writerow([8,element]) 