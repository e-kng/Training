from bs4 import BeautifulSoup
import requests
import pandas as pd
import pickle

r = requests.get("https://fr.wikipedia.org/wiki/Liste_de_groupes_de_black_metal")
data = r.text
soup = BeautifulSoup(data)

# Bands from wikipedia

band = []
country = []

for elt in soup.find_all('td'):
    text = elt.find('a').get('title') 
    if text is not None :
        if 'Drapeau' in text :
            text = text.split(' ')[-1]
            text = text.replace('l\'', '')
            country.append(text)
        else:
            text = text.split('(')[0]
            if text[-1] == ' ':
                text = text[:-1]
            band.append(text)

df = pd.DataFrame(band, country)
df.reset_index(inplace=True)
df.columns = ['country', 'band']
band_norway = df[df['country'] == 'Norvège']['band'].tolist()


# Discography from DarkLyrics

band_album = {}

for band in band_norway:
    if band == '1349':
        r = requests.get("http://www.darklyrics.com/"+'19'+"/"+band+".html")
    else:
        r = requests.get("http://www.darklyrics.com/"+band[0].lower()+"/"+band.lower().replace(' ','')+".html")
    
    data = r.text
    soup = BeautifulSoup(data)
    
    discog = []

    for elt in soup.find_all('h2'):
        
        if elt.find('strong') is not None :
            album = elt.find('strong').text
            album = album.replace('\"', '')
            discog.append(album)
        else:
            continue

    band_album[band] = discog


# Lyrics from DarkLyrics

band_lyrics = {}

for band in band_album.keys():
    lyrics_list = []

    for album in band_album[band]:
        #print(band,album)
        r = requests.get("http://www.darklyrics.com/lyrics/"+band.lower()+"/"+album.lower().replace(' ','')+".html")
        
        if r.status_code == 404:
            continue

        data = r.text
        soup = BeautifulSoup(data)

        try:
            for elt in soup.find_all('div',{'class':'lyrics'}):
                exclude = elt.find('div',{'class':'thanks'})
                exclude.extract()

                exclude = elt.find('div',{'class':'note'})
                exclude.extract()

                lyrics = elt.text
        except:
            continue

        title = []
        for elt in soup.find_all('h3'):
            title.append(elt.text)

        lyrics = lyrics.split('\n')

        for elt in lyrics :
            elt.replace('\r', '')

        for elt in lyrics :
            if elt in title :
                lyrics.remove(elt)

        exclusion = ['[Bonus Track]','[Ref:]', '[Chorus:]', '[instrumental]', '[Mayhem cover]', '', 'LYRICS']

        for elt in lyrics :
            if elt in exclusion :
                lyrics.remove(elt)
               
        lyrics_list = lyrics_list + lyrics

    lyrics_dict = {i:band for i in lyrics_list}

    band_lyrics.update(lyrics_dict)

file = open("black_metal_lyris.pkl", "wb")
pickle.dump(band_lyrics, file)
file.close()