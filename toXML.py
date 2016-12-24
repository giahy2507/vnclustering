__author__ = 'HyNguyen'



import os
import codecs
import xml.etree.ElementTree as ET



if __name__ == "__main__":

    data_path = "/Users/HyNguyen/Desktop/themes/cfvi-4th102016/clusters/20161004"
    clusters_dir = os.listdir(data_path)
    docs = []
    clusters_id = []
    titles = []

    for cluster_no, cluster_dir in enumerate(clusters_dir):
        if cluster_dir[0] == ".": continue
        cluster_path = data_path +"/"+ cluster_dir

        files_content = [filename for filename in os.listdir(cluster_path) if filename.find("content") !=-1 and filename.find("txt.tok") !=-1 and filename.find("tag")==-1 and filename[0]!="."]
        files_title = [filename for filename in os.listdir(cluster_path) if filename.find("title") !=-1 and filename.find("txt.tok") !=-1 and filename.find("tag")==-1 and filename[0]!="."]
        files_shortintro = [filename for filename in os.listdir(cluster_path) if filename.find("short_intro") !=-1 and filename.find("txt.tok") !=-1 and filename.find("tag")==-1 and filename[0]!="."]
        files_time = [filename for filename in os.listdir(cluster_path) if filename.find("time") !=-1 and filename.find("txt.tok") !=-1 and filename.find("tag")==-1 and filename[0]!="."]
        files_tag = [filename for filename in os.listdir(cluster_path) if filename.find("_tag.txt.tok") !=-1 and filename.find(".tag")==-1 and filename[0]!="."]
        files_link = [filename for filename in os.listdir(cluster_path) if filename.find("title") !=-1 and filename.find("tok")==-1 and filename[0]!="."]

        assert  len(files_content) == len(files_title) == len(files_shortintro) == len(files_time) == len(files_tag) == len(files_link)
        print("\n\n")
        print(cluster_dir,len(files_content))


        root = ET.Element("DATA")

        for cluster_id, (fn_content, fn_title, fn_intro, fn_time, fn_tag, fn_link) in enumerate(zip (files_content, files_title, files_shortintro, files_time, files_tag, files_link)):
            cluster_tag = ET.SubElement(root, "DOC")

            file_path = cluster_path + "/" + fn_title
            with codecs.open(file_path, mode="r", encoding="utf8") as f:
                title = f.readline()
                title_tag = ET.SubElement(cluster_tag, "TITLE")
                title_tag.text = title

            file_path = cluster_path + "/" + fn_intro
            with codecs.open(file_path, mode="r", encoding="utf8") as f:
                intro = f.read()
                intro_tag = ET.SubElement(cluster_tag, "INTRO")
                intro_tag.text = intro

            file_path = cluster_path + "/" + fn_content
            with codecs.open(file_path, mode="r", encoding="utf8") as f:
                content = f.read()
                content_tag = ET.SubElement(cluster_tag, "CONTENT")
                content_tag.text = content

            file_path = cluster_path + "/" + fn_time
            with codecs.open(file_path, mode="r", encoding="utf8") as f:
                time= f.readline()
                time_tag = ET.SubElement(cluster_tag, "TIME")
                time_tag.text = time

            file_path = cluster_path + "/" + fn_tag
            with codecs.open(file_path, mode="r", encoding="utf8") as f:
                tag = f.read()
                tag_tag = ET.SubElement(cluster_tag, "TAG")
                tag_tag.text = tag

            file_path = cluster_path + "/" + fn_link
            with codecs.open(file_path, mode="r", encoding="utf8") as f:
                link = " ".join(f.readlines()[1:])
                link_tag = ET.SubElement(cluster_tag, "LINK")
                link_tag.text = link


        ET.ElementTree(root).write("data/20161004/{0}.xml".format(cluster_dir),encoding="utf8")






