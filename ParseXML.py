__author__ = 'HyNguyen'


import xml.etree.ElementTree as ET
import os
import nltk


class MyDocument(object):

    def __init__(self, title = "", intro = "", content = "", time = "", tag = "", link = ""):
        if title is None: title = ""
        if intro is None: intro = ""
        if content is None: content = ""
        if time is None: time = ""
        if tag is None: tag = ""
        if link is None: link = ""
        self.title = title
        self.intro = intro
        self.content = content
        self.time = time
        self.tag = tag
        self.link = link

    def toTreeElement(self):
        doc_tag = ET.Element("DOC")
        title_tag = ET.SubElement(doc_tag, "TITLE")
        title_tag.text = self.title
        intro_tag = ET.SubElement(doc_tag, "INTRO")
        intro_tag.text = self.intro
        content_tag = ET.SubElement(doc_tag, "CONTENT")
        content_tag.text = self.content
        time_tag = ET.SubElement(doc_tag, "TIME")
        time_tag.text = self.time
        tag_tag = ET.SubElement(doc_tag, "TAG")
        tag_tag.text = self.tag
        link_tag = ET.SubElement(doc_tag, "LINK")
        link_tag.text = self.link
        return doc_tag

    def extract_SpecialNames(self, name):
        if name == "title":
            doc = self.title
        elif name == "intro":
            doc = self.intro
        elif name == "content":
            doc = self.content
        elif name == "all":
            doc = self.title + self.intro + self.content
        result = []
        for line in nltk.line_tokenize(doc):
            words = nltk.word_tokenize(line)
            if len(words) < 3:
                continue
            for word_id, word in enumerate(words):
                if word_id != 0  and word[0].isupper() and ( words[word_id-1] != "." or  words[word_id-1] != "..."):
                    result.append(word)
        return list(set(result))

    def extract_Quotes(self, name):
        if name == "title":
            doc = self.title
        elif name == "intro":
            doc = self.intro
        elif name == "content":
            doc = self.content
        elif name == "all":
            doc = self.title + self.intro + self.content
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





def convert(dir_in, dir_out):
    plct_dir = "data/phap-luat-chinh-tri"
    plct_dir_out = "data/save-data"
    clusters_file = [fn for fn in os.listdir(plct_dir) if fn[0]!= "."]
    for cluster_file in clusters_file:
        cluster_file_path = plct_dir + "/" + cluster_file
        cluster_file_out_path = plct_dir_out + "/" + cluster_file

        with open(cluster_file_path, encoding="utf8") as f:
            parser = ET.XMLParser(encoding="utf-8")
            root = ET.fromstring(f.read(), parser)
            out_root = ET.Element("CLUSTER")
            for idx in range(50):
                for cluster in root.findall('DOC_'+str(idx)):
                    title = cluster.find('TITLE').text
                    intro = cluster.find('INTRO').text
                    content = cluster.find('CONTENT').text
                    time = cluster.find('TIME').text
                    tag = cluster.find('TAG').text
                    link = cluster.find('LINK').text
                    mydoc = MyDocument(title, intro, content, time, tag, link)
                    doc_tag = mydoc.toTreeElement()
                    out_root.append(doc_tag)

            ET.ElementTree(out_root).write(plct_dir_out + "/{0}".format(cluster_file), encoding="utf8")




if __name__ == "__main__":

    plct_dir = "data/phap-luat-chinh-tri"
    clusters_file = [fn for fn in os.listdir(plct_dir) if fn[0]!= "."]
    docs = []
    for cluster_file in clusters_file[:5]:
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



    print("ttdt")
