__author__ = 'HyNguyen'
import numpy as np
import Levenshtein
import os
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np


def scale_0_1(matrix):
    maxeval = np.max(matrix)
    mineval = np.min(matrix)
    if maxeval == mineval:
        return np.zeros_like(matrix)
    matrix = (matrix - mineval)/(maxeval - mineval)
    for i in range(matrix.shape[0]):
        matrix[i][i]  = 0
    return matrix

def intersectionSet(a,b):
    return list( set(a) & set(b))

def sim_string(a,b):
    distance = Levenshtein.ratio(a, b)
    return distance

def train_tfidf():
    data_dir_1 = "/Users/HyNguyen/Documents/Research/Data/vn_express_1_tok"
    data_dir_2 = "/Users/HyNguyen/Documents/Research/Data/vn_express_2_tok"
    files_path = [data_dir_1 +"/" + fn for fn in os.listdir(data_dir_1) if fn.find(".tok") != -1] + [data_dir_2 +"/" + fn for fn in os.listdir(data_dir_2) if fn.find(".tok") != -1]
    # print(files_path)
    docs = []
    for file_path in files_path:
        with open(file_path, mode="r", encoding="utf8") as f:
            lines = f.readlines()
            for i in range(int(len(lines)/3)):
                doc = " ".join(lines[i*3 : i*3 + 3])
                docs.append(doc)

    # print(len(docs))

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,min_df=0.2,use_idf=True, ngram_range=(1,3))
    tfidf_vectorizer.fit(docs)
    with open("model/TFIDF_model.pickle", mode="wb") as f:
        pickle.dump(tfidf_vectorizer,f)
