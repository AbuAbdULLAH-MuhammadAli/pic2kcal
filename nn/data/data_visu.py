import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from split import filter_outliers


def bar_plot(data, attribute, xlabel, ylabel, title):
    ls = []

    for idx, row in df.iterrows():
        ls.append(len(row[attribute]))

    counter = Counter(ls)
    # extract frequencies as values
    dictlist = []
    for key in counter:
        value = counter[key]
        temp = [key, value]
        dictlist.append(temp)

    # sort
    sorted_dictlist = sorted(dictlist, key=lambda x: x[0])

    # extract to separate lists
    num, frequencies = map(list, zip(*sorted_dictlist))

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.bar(num, frequencies)

    plt.show()


def kcal_plot(data, attribute, xlabel, ylabel, title):
    ls = data[attribute]
    print(ls)

    # extract frequencies as values
    counter = Counter(ls)
    dictlist = []
    for key in counter:
        value = counter[key]
        temp = [key, value]
        dictlist.append(temp)

    # sort
    sorted_dictlist = sorted(dictlist, key=lambda x: x[0])

    # extract to separate lists
    num, frequencies = map(list, zip(*sorted_dictlist))

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.set_yscale('log')
    ax.scatter(num, frequencies)

    plt.show()


def kcal_hist(data, attribute, xlabel, ylabel, title):
    ls = list(data[attribute])

    bef_count = len(ls)
    ls, filt_min, filt_max = filter_outliers(ls)
    print(f"filtering kcal to [{filt_min}, {filt_max}]")

    print(f"removed {bef_count - len(ls)} of {bef_count}")
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.set_yscale('log')
    ax.hist(ls, bins=100)

    plt.show()


if __name__ == "__main__":
    # read json
    with open("per_portion_data.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # bar plot of image frequencies
    # bar_plot(df, attribute='picture_files', xlabel='# images per recipe',
    #          ylabel='Frequency', title='Number of Images per Recipe')

    # plot of kcal frequencies
    kcal_hist(
        df,
        attribute="kcal_per_portion",
        xlabel="kcal per recipe-portion",
        ylabel="Frequency",
        title="Kcal per Recipe",
    )

