from bs4 import BeautifulSoup as BS
import requests

def main():
    #download_file("http://mensenhandel.nl/files/pdftest2.pdf")
    response = requests.get("http://www.archiviolastampa.it/component/option,com_lastampa/task,search/mod,avanzata/action,viewer/Itemid,3/page,1/articleid,1282_01_1867_0002_0001_18769532/anews,true/")
    soup = BS(response.text)
    print(soup)
    div = soup.find("div",{"class": "maintabletemplate"})
    tds = div.find_all('td')
    prefix = "https://ocw.mit.edu"
    for each in tds[1::2]:
    	try:
    		#print(each.find("a").string,prefix+each.find("a")["href"])
    		#print(each.find("a").string)
    		name = each.find("a").string
    		if name!=None:
    			download_file(prefix+each.find("a")["href"],name)
    		else:
    			download_file(prefix+each.find("a")["href"],'note')
    	except:
    		continue

def download_file(download_url,name):
    response = requests.get(download_url)
    with open(name+'.pdf', 'wb') as f:
    	f.write(response.content)
    print("Completed")

if __name__ == "__main__":
	#download_file("https://ocw.mit.edu/courses/mathematics/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/lecture-notes/MIT18_S096F13_lecnote1.pdf")
	main()