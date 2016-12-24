__author__ = 'HyNguyen'
import numpy as np
from ParseXML import MyDocument
from utilities import *
import xml.etree.ElementTree as ET
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def get_SimMatrix_TRComparer_SpecialName(docs):

    docs_SpecialName = []
    for doc in docs:
        docs_SpecialName.append(doc.extract_SpecialNames("all"))

    SN_matrix = np.zeros((len(docs), len(docs)),dtype=np.float32)
    for i in range(0,len(docs)):
        for j in range(i+1,len(docs)):
            tu = (len (intersectionSet(docs_SpecialName[i], docs_SpecialName[j]))  * 1.0)
            mau = ( np.log(len(docs_SpecialName[i])+1) + np.log(len(docs_SpecialName[j])+1) )
            SN_matrix[i][j] = tu/mau
            if np.isnan(SN_matrix[i][j]) or np.isinf(SN_matrix[i][j]):
                SN_matrix[i][j] = 0
                continue
            SN_matrix[j][i] = SN_matrix[i][j]

    return scale_0_1(SN_matrix)


def get_SimMatrix_TRComparer_Quotes(docs):
    docs_Quotes = []
    for doc in docs:
        docs_Quotes.append(doc.extract_Quotes("all"))

    Q_matrix = np.zeros((len(docs), len(docs)),dtype=np.float32)
    for i in range(0,len(docs_Quotes)):
        for j in range(i+1,len(docs_Quotes)):
            count = 0
            for quote1 in docs_Quotes[i]:
                for quote2 in docs_Quotes[j]:
                    if sim_string(quote1, quote2) > 0.8:
                        count +=1
            if count == 0:
                continue
            Q_matrix[i][j] = (count*1.0)/(np.log(len(docs_Quotes[i])+1)+np.log(len(docs_Quotes[j])+1))
            if np.isnan(Q_matrix[i][j]) or np.isinf(Q_matrix[i][j]):
                Q_matrix[i][j] = 0
                continue
            Q_matrix[j][i] = Q_matrix[i][j]

    return scale_0_1(Q_matrix)

def get_SimMatrix_TFIDF(docs):
    docs_str = []
    for doc in docs:
        docs_str.append(doc.title + doc.intro + doc.content)
    with open("model/TFIDF_model.pickle", mode="rb") as f:
        tfidf_vectorizer = pickle.load(f)
    docs_vec = tfidf_vectorizer.transform(docs_str)
    sim_matrix = cosine_similarity(docs_vec)
    return scale_0_1(sim_matrix)



if __name__ == "__main__":
    plct_dir = "data/phap-luat-chinh-tri"
    clusters_file = [fn for fn in os.listdir(plct_dir) if fn[0]!= "."]
    docs = []
    for cluster_file in clusters_file:
        cluster_file_path = plct_dir + "/" + cluster_file
        with open(cluster_file_path, encoding="utf8") as f:
            parser = ET.XMLParser(encoding="utf-8")
            root = ET.fromstring(f.read(), parser)
            for cluster in root.findall('DOC'):
                title = cluster.find('TITLE').text
                intro = cluster.find('INTRO').text
                content = cluster.find('CONTENT').text
                time = cluster.find('TIME').text
                tag = cluster.find('TAG').text
                link = cluster.find('LINK').text
                mydoc = MyDocument(title, intro, content, time, tag, link)
                docs.append(mydoc)

    print(len(docs))

    SN_matrix = get_SimMatrix_TRComparer_SpecialName(docs)
    Q_matrix = get_SimMatrix_TRComparer_Quotes(docs)


    TFIDF_matrix = get_SimMatrix_TFIDF(docs)

    print(SN_matrix.shape)
    print(Q_matrix.shape)
    print(TFIDF_matrix.shape)
