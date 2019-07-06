import json
import os

import pandas as pd
from bs4 import BeautifulSoup
import re

"""
extracts info for each food item from fddb
requires: 

TODO: set directory 'DIR' in __main__
"""


def extractor_standard(string):

    # print(string)
    # split text at different entries
    naehrwert_idx = string.find('Nährwerte für')
    end_idx = string.find('Angaben korrigieren')

    # workaround if 'Angaben korrigieren' is not present
    # if end_idx > naehrwert_idx:
    #

    # consider string from naehrwerte_idx on
    # string = string[naehrwert_idx:end_idx]
    string = string[naehrwert_idx:end_idx]

    # define end index_
    # end_idx =

    print(string)
    # string = string[0:]
    string = string[0:end_idx]


    # print('STRING: ', string)
    brennwert_idx = string.find('Brennwert')
    kalorien_idx = string.find('Kalorien')
    protein_idx = string.find('Protein')
    ballasstoff_idx = string.find('Ballaststoffe')
    kohlenhydrate_idx = string.find('Kohlenhydrate')
    davon_zucker_idx = string.find('davon Zucker')
    if davon_zucker_idx == -1:
        davon_zucker_idx = kohlenhydrate_idx

    fett_idx = string.find('Fett')
    broteinheiten_idx = string.find('Broteinheiten')
    cholesterin_idx = string.find('Cholesterin')

    indices = [naehrwert_idx, brennwert_idx, kalorien_idx, protein_idx, kohlenhydrate_idx, davon_zucker_idx,
               fett_idx, ballasstoff_idx, broteinheiten_idx, cholesterin_idx]

    # separate parts and remove double spaces
    parts = [re.sub(' +', ' ', string[i:j]) for i, j in zip(indices, indices[1:] + [None])]

    # only keep reelvant parts
    parts = parts[0:11]
    part_dict = {}

    for i in range(0, len(parts)):
        part = parts[i]
        if part.strip() != '':
            # remove double-dot and all whitespace
            part = re.sub(r"[\s:]*", "", part)

            if 'Cholesterin' in part:
                # find index of mg
                mg_idx = part.find('mg')
                part = part[:mg_idx+2]

            # extract letters / words
            naehrwert, *einheit = list(filter(None, re.split(r'[(-?\d+\,?.?\d)]', part)))
            # extract numbers
            menge = list(filter(None, re.split(r'[a-z]|[A-Z]', part)))
            # transform list of strings to strings
            menge = ''.join(map(str, menge))
            einheit = ''.join(map(str, einheit))
            if 'Cholesterin' in part:
                einheit = einheit[:2]
            # save dict
            part_dict[naehrwert] = {'Menge': menge, 'Einheit': einheit}

    dict = {re.sub(' +', ' ', string[0:brennwert_idx]): part_dict}
    return dict


def extractor_specialized(dict, idx):
    part_dict = {}

    if idx == 0:
        # find all entries in the first specialized table
        specialized_table_entries = dict.find_all('div', {'style': "padding:0px 0px 2px 0px;"})
        for part in specialized_table_entries:
            part_text = part.get_text()

            # split part_texts
            if part_text != '':
                # remove double-dot and all whitespace
                part_text = re.sub(r"[\s:]*", "", part_text)

                # extract letters / words
                naehrwert, *einheit = list(filter(None, re.split(r'[(-?\d+\,?.?\d)]', part_text)))

                # extract numbers
                menge = list(filter(None, re.split(r'[a-z]|[A-Z]', part_text)))
                # transform list of strings to strings
                menge = ''.join(map(str, menge))
                einheit = ''.join(map(str, einheit))

                # save dict
                part_dict[naehrwert] = {'Menge': menge, 'Einheit': einheit}

    elif idx == 1:
        specialized_table_entries = dict.find_all('p')
        for p in specialized_table_entries:
            p_text = p.get_text()
            if 'Brennwert' in p_text:
                # get indices
                brennwert_idx = p_text.find('Brennwert')
                kalorien_idx = p_text.find('Kalorien')

                indices = [brennwert_idx, kalorien_idx]

                # separate parts and remove double spaces
                sub_parts = [re.sub(' +', ' ', p_text[i:j]) for i, j in zip(indices, indices[1:] + [None])]

                for i in range(0, len(sub_parts)):
                    sub_part = sub_parts[i]
                    if sub_part != '':
                        # remove double-dot and all whitespace
                        sub_part = re.sub(r"[\s:]*", "", sub_part)

                        # extract letters / words
                        naehrwert, *einheit = filter(None, re.split(r"[(-?\d+\,?.?\d)]", sub_part))
                        menge = list(filter(None, re.split(r'[a-z]|[A-Z]', sub_part)))
                        # transform list of strings to strings
                        menge = ''.join(map(str, menge))
                        einheit = ''.join(map(str, einheit))

                        # save dict
                        part_dict[naehrwert] = {'Menge': menge, 'Einheit': einheit}

    return part_dict

def normalize_spaces(s):
    return re.sub(r" +", " ", s)

def extract_from_html(data, file_name, folder_name):
    soup = BeautifulSoup(open(file_name, 'rb'), 'html.parser')

    # extract the product name
    product_name = soup.find('h1').get_text()

    for tag in soup.find_all("meta"):
        if tag.get("name", None) == "title":
            b = tag.get("content", None)
    product_name = b[:-32]

    # extract the source text
    source = ''
    producer = ''
    food_category = ''
    p_tags = soup.find_all('p')
    for tag in p_tags:
        tag_text = tag.get_text()
        if 'Produktgruppe' in tag_text:
            food_category = tag_text.replace('Produktgruppe: ', '')
        if 'Hersteller' in tag_text:
            producer = tag.text.replace('Hersteller: ', '')
        if 'Datenquelle' in tag_text:
            source = tag_text.replace('Datenquelle: ', '')

    # dictionary for saving
    # - the folder name as id
    # - the food category
    # - the producer
    # - the source of the specification
    # - all information about nutrients in the different panels
    dict = {}

    # save the folder name as id
    dict['name'] = soup.find('h1').get_text()  # TODO folder_name
    dict['Lebensmittelgruppe'] = food_category
    dict['Hersteller'] = producer
    dict['Quelle'] = source

    ''' Standard Zutaten (fuer 100g / 100ml) '''
    standard_content = soup.find_all('div', {'class': 'standardcontent'})
    standard_dict = {}

    for i in standard_content:
        itext = i.get_text()
        if 'Nährwerte' in itext:
            standard_dict = {}
            # extract info to dict
            # standard_dict = extractor_standard(itext)
            for div in soup.select("div.standardcontent > div:nth-child(2) > div"):
                if "itemsec2012" in div.get("class", []):
                    cur = {}
                    standard_dict[normalize_spaces(div.get_text())] = cur
                else:
                    if div.get("id") == "chartcontainer":
                        continue
                    if list(div.children)[0].name == "table":
                        continue
                    k, v = div.children

                    naehrwert, einheit = re.match(r"^(\d+(?:,\d+)?%?)(?: (\w+))?$", v.get_text()).groups()
                    naehrwert = naehrwert.replace(",", ".")
                    if naehrwert[-1] == "%":
                        naehrwert = float(naehrwert[:-1])
                        if einheit is not None:
                            raise Exception("HEEEEEEEEE")
                        einheit = "%"
                    else:
                        naehrwert = float(naehrwert)
                    cur[normalize_spaces(k.get_text())] = {'Menge': naehrwert, 'Einheit': einheit or ""}


        else:
            raise Exception('Keine Informationen über Standard-Nährwerte enthalten.')
    dict['Standard Nährwerte'] = standard_dict

    ''' Werte für spezialisierte Angabe (zB Dose von 250g) - 
    hiervon liegen mehrere Eintraege vor '''
    specialized_table = soup.find_all('div', {'class': 'serva'})

    # find headers
    specialized_table_headers = []
    for spec in specialized_table:
        specialized_table_headers.append(spec.find('a').get_text())

    specialized_dict = {}

    # specialized table 0
    specialized_dict_0 = {}
    specialized_dict_0 = extractor_specialized(specialized_table[0], idx=0)

    # further specialized tables
    if (len(specialized_table) > 1):
        for i in range(1, len(specialized_table)):
            specialized_dict_1 = {}
            specialized_dict_1 = extractor_specialized(specialized_table[i], idx=1)

            specialized_dict[specialized_table_headers[i]] = specialized_dict_1

    # number of ratings
    num_ratings = 0
    hrefs = soup.find_all('a', attrs={'href': re.compile("^https://")})
    for href in hrefs:
        if '#bewertungen' in str(href):
            s = href.get_text()
            num_ratings = ''.join(i for i in s if i.isdigit())

    # extract id
    id = soup.select(".breadcrumb > a")[-1].get("href")

    # save to dict
    dict['Spezifische Nährwerte'] = {specialized_table_headers[0]: specialized_dict_0}
    dict['Spezifische Nährwerte'].update(specialized_dict)
    dict['Bewertungen'] = num_ratings
    dict['Id'] = id

    # save all to data under the current product_name
    data[product_name] = dict
    return data, product_name


def getKeys(D, answer):
    if not D:
        return
    else:
        for k in D:
            answer.append(D[k])
        return answer


if __name__ == '__main__':

    DIR = "/home/veheusser/Code_Projects/cvhci_praktikum/fddb/"

    DATA_DIR = DIR + "/fddb.info/db/de/lebensmittel/"

    os.chdir(DATA_DIR)

    # create dictionary
    data = {}
    b = 0

    # list of products
    product_names = []

    for folder in os.listdir(DATA_DIR):
        if not (folder == 'selbst_gemacht_100_kalorien_dummy'):
            # print('current folder: ', folder, ' ', b)

            if True: #folder == "naturprodukt_banane_roh":#"rewe_kartoffeln":

                print('folder: ', folder)

                b += 1
                os.chdir(DATA_DIR)

                # only handle index.html - files
                for file in os.listdir(folder):
                    print('folder: ', folder)
                    if file == 'index.html':
                        os.chdir(folder)
                        data, product_name = extract_from_html(data, file, folder)
                        product_names.append(product_name)

    # to array of dicts
    data = getKeys(data, answer=[])

    # save data to json file
    os.chdir(DIR)
    with open('fddb_data.json', 'w') as outfile:
        # saved as pretty print with indent=4, sort_Keys=True
        json.dump(data, outfile, ensure_ascii=False, indent=4) # , sort_keys=True)
    outfile.close()

    # save list of product names to csv file
    df = pd.DataFrame(product_names)
    df.to_csv('fddb_product_names.csv', index=False, header=False)
