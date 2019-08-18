from bs4 import BeautifulSoup
import os
import json
import pandas as pd


"""
extracts json holding informations about each food item from a number of folders from 
the DEBInet nutrition specification.

TODO: set directory 'DIR' in __main__
"""

def extract_from_html(data, file_name, folder_name):
    soup = BeautifulSoup(open(file_name), "html.parser")

    # extract the product name
    product_name = soup.find('h1').get_text()

    # extract the food category
    food_category = soup.find('em').get_text()

    # extract the source text
    source = ''
    p_tags = soup.find_all('p')
    for tag in p_tags:
        tag_text = tag.get_text()
        if 'Nährwertangaben' in tag_text:
            source = tag_text

    # dictionary for saving
    # - the folder name as id
    # - the food category
    # - the source of the specification
    # - all information about nutrients in the different panels
    dict = {}

    # save the folder name as id
    dict['Id'] = folder_name
    dict['Lebensmittelgruppe'] = food_category
    dict['Quelle'] = source

    # get all panels containing nutrient information
    all_nutrients = soup.find_all('div', {'class': 'panel panel-default center-block'})

    # extract text from all panels
    for i in range(0, len(all_nutrients)):
        # header of current panel
        header = all_nutrients[i].find('h3').get_text()

        # dictionary for saving all information from the current
        # tr-table containing specific nutrients
        current_table_dict = {}

        # find all tables
        table = all_nutrients[i].find_all('tr')

        for table_entry in table:
            # consider all entries but the table header
            if not(table_entry.find('th')):
                lines = table_entry.find_all('td')

                if header == 'Allergene und Zusatzstoffe':
                    current_table_dict[lines[0].get_text()] = {'E-Nummer': lines[1].get_text()}
                else:
                    current_table_dict[lines[0].get_text()] = {'Menge': lines[1].get_text(),
                                                               'Einheit': lines[2].get_text()}

        # save information from the current tr-table in the dictionary under the current header
        dict[header] = current_table_dict

    # save all to data under the current product_name
    data[product_name] = dict

    return data, product_name


if __name__ == '__main__':
    DIR = "/home/veheusser/Code_Projects/cvhci_praktikum/preprocessing/"

    DATA_DIR = DIR+"hi/de/"
    os.chdir(DATA_DIR)

    # create the dictionary
    data = {}

    # list of products
    product_names = []

    for file in os.listdir(DATA_DIR):
        print('current folder: ', file)


        # ignore html files. Only handle html files in subfolders
        if not(file.endswith('.html')):
            subfolder = DATA_DIR+file
            os.chdir(subfolder)
            for html_file in os.listdir(subfolder):
                # extract information from the current html file
                data, product_name = extract_from_html(data, html_file, file)
                product_names.append(product_name)

    # save data to json file
    os.chdir(DIR)
    with open('nutrient_data.json', 'w') as outfile:
        # saved as pretty print with indent=4, sort_Keys=True
        json.dump(data, outfile, ensure_ascii=False, indent=4, sort_keys=True)
    outfile.close()

    # save list of product names to csv file
    df = pd.DataFrame(product_names)
    df.to_csv('product_names.csv', index=False, header=False)