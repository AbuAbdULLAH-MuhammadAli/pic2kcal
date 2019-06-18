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

    # split text at different entries
    naehrwert_idx = string.find('Nährwerte')
    end_idx = string.find('Angaben korrigieren')
    # consider string from naehrwerte_idx on
    string = string[naehrwert_idx:end_idx]

    brennwert_idx = string.find('Brennwert')
    kalorien_idx = string.find('Kalorien')
    protein_idx = string.find('Protein')
    ballasstoff_idx = string.find('Ballaststoffe')
    kohlenhydrate_idx = string.find('Kohlenhydrate')
    davon_zucker_idx = string.find('davon Zucker')
    fett_idx = string.find('Fett')
    broteinheiten_idx = string.find('Broteinheiten')


    indices = [naehrwert_idx, brennwert_idx, kalorien_idx, protein_idx, kohlenhydrate_idx, davon_zucker_idx,
               fett_idx, ballasstoff_idx, broteinheiten_idx] #, end_idx]

    # separate parts and remove double spaces
    parts = [re.sub(' +', ' ', string[i:j]) for i, j in zip(indices, indices[1:] + [None])]

    # only keep reelvant parts
    parts = parts[0:9]
    part_dict = {}

    for i in range(1, len(parts)):
        part = parts[i]
        if part != '':
            # remove double-dot and all whitespace
            part = re.sub(r"[\s:]*", "", part)

            # extract letters / words
            naehrwert, *einheit = list(filter(None, re.split(r'[(-?\d+\,?.?\d)]', part)))
            # extract numbers
            menge = list(filter(None, re.split(r'[a-z]|[A-Z]', part)))
            # transform list of strings to strings
            menge = ''.join(map(str, menge))
            einheit = ''.join(map(str, einheit))

            # save dict
            part_dict[naehrwert] = {'Menge': menge, 'Einheit': einheit}

    dict = {parts[0]: part_dict}
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
                part_dict[naehrwert] = {'Menge': menge[0], 'Einheit': einheit}

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
                        naehrwert, *einheit = list(filter(None, re.split(r'[(-?\d+\,?.?\d)]', sub_part)))
                        # extract numbers
                        menge = list(filter(None, re.split(r'[a-z]|[A-Z]', sub_part)))
                        # transform list of strings to strings
                        menge = ''.join(map(str, menge))
                        einheit = ''.join(map(str, einheit))

                        # save dict
                        part_dict[naehrwert] = {'Menge': menge, 'Einheit': einheit}

    return part_dict


def extract_from_html(data, file_name, folder_name):
    soup = BeautifulSoup(open(file_name, 'rb'), 'html.parser')

    # extract the product name
    product_name = soup.find('h1').get_text()

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
            # extract info to dict
            standard_dict = extractor_standard(itext)
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
    if (len(specialized_table)>1):
        for i in range(1, len(specialized_table)):
            specialized_dict_1 = {}
            specialized_dict_1 = extractor_specialized(specialized_table[i], idx=1)

            specialized_dict[specialized_table_headers[i]] = specialized_dict_1

    # save to dict
    dict['Spezifische Nährwerte'] = {specialized_table_headers[0]: specialized_dict_0}
    dict['Spezifische Nährwerte'].update(specialized_dict)

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
    b=0
    DIR = "/home/veheusser/Code_Projects/cvhci_praktikum/fddb/"

    DATA_DIR = DIR+"/fddb.info/db/de/lebensmittel/"
    os.chdir(DATA_DIR)

    # create dictionary
    data = {}

    # list of products
    product_names = []

    for folder in os.listdir(DATA_DIR):
        if not(folder == 'selbst_gemacht_100_kalorien_dummy'):
            b+=1
            print('current folder: ', folder, ' ', b)
            os.chdir(DATA_DIR)

            # only handle index.html - files
            for file in os.listdir(folder):
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
        json.dump(data, outfile, ensure_ascii=False, indent=4, sort_keys=True)
    outfile.close()

    # save list of product names to csv file
    df = pd.DataFrame(product_names)
    df.to_csv('fddb_product_names.csv', index=False, header=False)