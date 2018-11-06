# packages required

import numpy
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# build a classifier

def create_clf(v):
    data = pandas.read_csv("data_set.txt", sep = "\t", names = ['Status', 'Message'])
    counts = v.fit_transform(data['Message'].values)
    clf = MultinomialNB()
    targets = data['Status'].values
    return clf.fit(counts, targets)

# take email input

def email_input():
    vect = CountVectorizer()
    c = create_clf(vect)
    i = input("enter your email: ")
    s = [i]
    test = vect.transform(s)
    res = c.predict(test)
    print("This email is marked as", res[0])
    ask_conf(i)

# update the data set with new entries

def update_dataset(status, message):
    data_file = open("data_set.txt", "a+")
    data_file.write("\n")
    data_file.write(status)
    data_file.write("\t")
    data_file.write(message)

# ask user confirmation before updating the data set

def ask_conf(string):
    print("Mail details:", string)
    print("Was the email a spam? [1 for yes / 0 for no]: ", end = '')
    conf = int(input())
    if conf == 1:
        label = "spam"
        update_dataset(label, string)
    elif conf == 0:
        label = "ham"
        update_dataset(label, string)
    else:
        print("Invalid input!")
        ask_conf(string)

# main function

email_input()