from bs4 import BeautifulSoup
import os
import json


"""
extracts json tabular from html
"""


def extract_from_html(data, filename):
    # print('##### current filename: ', filename)
    # print('##### current wd: ', os.getcwd())

    soup = BeautifulSoup(open(filename), "html.parser")
    table_tag = soup.table
    print(table_tag.findall('h1'))
    # print(table_tag)
    # add name of current item to data
    # print(table_tag['h1'])

    # if (table_tag['class'] == ['Hauptnährstoffe'] or
    #         table_tag['class'] == ['Vitamine'] or
    #         table_tag['class'] == ['Mineralstoffe und Spurenelemente'] or
    #         table_tag['class'] ==
    # ):
    # if (table_tag['class'] == ['table-responsive']):
    #     print('SUCCESS')
    #     # print(table_tag.tr.th.get_text() + " " + table_tag.tr.td.get_text())
    #     a = table_tag.next_sibling
    #     # print(a)
    #     # print(table_tag.contents)
    #
    # else:
    #     print('\nFAIL')

    return data


if __name__ == '__main__':

    DIR = "/home/veheusser/Code_Projects/pic2kcal/preprocessing/hi/de/"
    os.chdir(DIR)
    data = {}
    a = 1

    for file in os.listdir(DIR):
        while a < 3:
            print('i am here: ', file)
            os.chdir(DIR)

            # ignore html files. Only handle subfolders
            if not(file.endswith('.html')):
                subfolder = DIR+file
                os.chdir(subfolder)
                for file in os.listdir(subfolder):
                    data = extract_from_html(data, file)

            a+=1

        json_data = json.dumps(data)
