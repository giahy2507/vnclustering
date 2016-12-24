__author__ = 'HyNguyen'

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MeanShift, DBSCAN ,AgglomerativeClustering, AffinityPropagation
import os
import codecs
import numpy as np
import Pycluster
import nltk
import Levenshtein
import panda as pd


from sklearn import metrics


def scale_0_1(matrix):
    maxeval = np.max(matrix)
    mineval = np.min(matrix)
    if maxeval == mineval:
        return np.zeros_like(matrix)
    matrix = (matrix - mineval)/(maxeval - mineval)
    return matrix

def intersectionSet(a,b):
    return list( set(a) & set(b))

def sim_string(a,b):
    distance = Levenshtein.ratio(a, b)
    return distance



def get_No_SpecialName(doc1 = "", doc2 = ""):
    doc1_SN = extract_SpecialName(doc1)
    doc2_SN = extract_SpecialName(doc2)
    return len(intersectionSet(doc1_SN, doc2_SN))

def extract_SpecialName(doc = "" ):
    result = []
    for line in nltk.line_tokenize(doc):
        words = nltk.word_tokenize(line)
        if len(words) < 3:
            continue
        for word_id, word in enumerate(words):
            if word_id != 0  and word[0].isupper() and ( words[word_id-1] != "." or  words[word_id-1] != "..."):
                result.append(word)
    return list(set(result))

def get_No_Quotes(doc1 = "", doc2 = ""):
    doc1_Q = extract_Quotes(doc1)
    doc2_Q = extract_Quotes(doc2)
    count = 0
    for quote1 in doc1_Q:
        for quote2 in doc2_Q:
            if sim_string(quote1, quote2) > 0.8:
                count +=1
    return count
def extract_Quotes(doc= ""):
    stack = []
    result = []
    flag = False

    words = nltk.word_tokenize(doc)

    for word_id, word in enumerate(words):
        if word == "\"" or word=="\"\"" or word == "''" or word == "'" or word =="``" or word =="`":
            if flag == True: flag = False
            else: flag = True
        else:
            if flag == True: stack.append(word)
            elif flag == False:
                if len(stack) > 0:
                    result.append(" ".join(stack).lower())
                    stack = []

    if len(stack) > 0:
        result.append(" ".join(stack).lower())
    return result

def get_SimMatrix_No_SpecialName(docs):

    docs_SpecialName = []
    for doc in docs:
        docs_SpecialName.append(extract_SpecialName(doc))


    SN_matrix = np.zeros((len(docs), len(docs)),dtype=np.float32)
    for i in range(0,len(docs)):
        for j in range(i+1,len(docs)):
            SN_matrix[i][j] = len(intersectionSet(docs_SpecialName[i], docs_SpecialName[j]))
            SN_matrix[j][i] = SN_matrix[i][j]
    return scale_0_1(SN_matrix)


def get_SimMatrix_No_Quotes(docs):
    SN_matrix = np.zeros((len(docs), len(docs)),dtype=np.float32)
    for i in range(0,len(docs)):
        for j in range(i+1,len(docs)):
            SN_matrix[i][j] = get_No_Quotes(docs[i],docs[j])
            SN_matrix[j][i] = SN_matrix[i][j]
    SN_matrix = scale_0_1(SN_matrix)
    SN_matrix_idx = SN_matrix == 0
    SN_matrix[SN_matrix_idx] = 0.0001
    return SN_matrix

def get_SimMatrix_No_Quotes2(docs):
    SN_matrix = np.zeros((len(docs), len(docs)),dtype=np.float32)
    for i in range(0,len(docs)):
        for j in range(i+1,len(docs)):
            SN_matrix[i][j] = get_No_Quotes(docs[i],docs[j])
            SN_matrix[j][i] = SN_matrix[i][j]
    return SN_matrix


# TRComparer
def get_TRComparer_ScpecialName(doc1 = "", doc2 = ""):
    doc1_SN = extract_SpecialName(doc1)
    doc2_SN = extract_SpecialName(doc2)
    return (len(intersectionSet(doc1_SN, doc2_SN))*1.0)/(np.log(len(doc1_SN)+1)+np.log(len(doc2_SN)+1))
def get_TRComparer_Quotes(doc1 = "", doc2 = ""):
    doc1_Q = extract_Quotes(doc1)
    doc2_Q = extract_Quotes(doc2)
    count = 0
    for quote1 in doc1_Q:
        for quote2 in doc2_Q:
            if sim_string(quote1, quote2) > 0.8:
                count +=1
    return (count*1.0)/(np.log(len(doc1_Q)+1)+np.log(len(doc2_Q)+1))

def get_SimMatrix_TRComparer_SpecialName(docs):

    docs_SpecialName = []
    for doc in docs:
        docs_SpecialName.append(extract_SpecialName(doc))

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
    SN_matrix = np.zeros((len(docs), len(docs)),dtype=np.float32)
    for i in range(0,len(docs)):
        for j in range(i+1,len(docs)):
            SN_matrix[i][j] = get_TRComparer_Quotes(docs[i],docs[j])
            SN_matrix[j][i] = SN_matrix[i][j]
    return scale_0_1(SN_matrix)




import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS

def visualize(dist, clusters, titles):

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)
    xs, ys = pos[:, 0], pos[:, 1]

    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#ff0000', 6: '#ffd700', 7:'#000000', 8:'#0000ff' , 9:'#00ffff'}
    cluster_names = {}

    for i in range(9):
        cluster_names[i] = str(i)


    #some ipython magic to show the matplotlib plots inline
    # %matplotlib inline

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

    #group by cluster
    groups = df.groupby('label')


    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  #show legend with only 1 point

    #add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)



plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)


if __name__ == "__main__":


    data_path = "/Users/HyNguyen/Desktop/themes/cfvi-4th102016/clusters/20161004"
    clusters_dir = os.listdir(data_path)
    docs = []
    clusters_id = []
    titles = []
    for cluster_no, cluster_dir in enumerate(clusters_dir[:10]):
        if cluster_dir[0] == ".": continue
        cluster_path = data_path +"/"+ cluster_dir
        files_content = [filename for filename in os.listdir(cluster_path) if filename.find("content") !=-1 and filename.find("txt.tok") !=-1 and filename.find("tag")==-1 and filename[0]!="."]
        files_title = [filename for filename in os.listdir(cluster_path) if filename.find("title") !=-1 and filename.find("txt.tok") !=-1 and filename.find("tag")==-1 and filename[0]!="."]
        files_shortintro = [filename for filename in os.listdir(cluster_path) if filename.find("short_intro") !=-1 and filename.find("txt.tok") !=-1 and filename.find("tag")==-1 and filename[0]!="."]
        assert  len(files_content) == len(files_title) == len(files_shortintro)
        print(cluster_dir,len(files_content))
        for fn_content, fn_title, fn_intro in zip (files_content, files_title, files_shortintro):
            doc = ""
            file_path = cluster_path + "/" + fn_title
            with codecs.open(file_path, mode="r", encoding="utf8") as f:
                doc += f.readline()
                titles.append(f.readline())

            file_path = cluster_path + "/" + fn_content
            with codecs.open(file_path, mode="r", encoding="utf8") as f:
                doc += f.read()

            if cluster_no == 5 or cluster_no == 6 or cluster_no == 9:
                print(cluster_no)
            docs.append(doc)
            names = extract_SpecialName(doc)
            quotes = extract_Quotes(doc)

            print("Name",names)
            print("Quote",quotes)
            print ("cluster_no: " + str(cluster_no))
            print (doc + "\n\n")

            clusters_id.append(cluster_no)


    TRSNSimMatrix  = get_SimMatrix_No_SpecialName(docs)
    # TRQSimMatrix = get_SimMatrix_No_Quotes(docs)
    ahihi  =   TRSNSimMatrix
    scalele = scale_0_1( ahihi )
    print(ahihi)
    print(scalele)


    #
    # scalele = get_SimMatrix_No_Quotes2(docs)
    # print(scalele)

    mydbscan = AffinityPropagation(affinity="precomputed")
    ahihi2 = mydbscan.fit_predict(scalele)
    print("True")
    print(clusters_id)
    print("Predict")
    print(list(ahihi2))

    print(metrics.adjusted_rand_score(clusters_id, ahihi2))


    visualize(1-ahihi, clusters_id, titles)

    # label_true =    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9]
    # label_predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 3, 4, 3, 4, 4, 4, 4, 3, 3, 4, 5, 5, 5, 5, 2, 2, 2, 4, 2, 2, 2]
    # print(metrics.adjusted_rand_score(label_true, label_predict))