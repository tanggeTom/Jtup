import json
import re
import string
import numpy as np
import jieba
from gensim.models import Word2Vec
import os
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer  # WordNetLemmatizer


# Update the new feature code semantics
def get_project_in_json(projectpath, projectname):
    gumtree_path = "E:\\project_collection\\python_entropy\\newold"
    model = Word2Vec.load("word2vec_w_5.model")
    reg = "([a-z])([A-Z])"  # Matches the hump of case exchange
    lemmatizer = WordNetLemmatizer()
    for index, json_file in enumerate(os.listdir(projectpath)):
        old_wv = np.zeros(shape=model.wv["a"].shape)  # 初始化
        new_wv = np.zeros(shape=model.wv["a"].shape)
        for sequence in ["old", "new"]:
            try:
                with open(os.path.join(gumtree_path, projectname, sequence, json_file.replace(".json", "") + ".java"),
                          "r") as f:
                    class_content = f.read().replace("\n", "")
                    for i in string.punctuation:
                        class_content = class_content.replace("{}".format(i), " ")
                    result = " ".join(jieba.cut(class_content))
                    result = [i for i in result.split(" ") if i != ""]  # The list stores all the words
                    count_word_nofind = 0
                    total_vector = np.zeros(shape=model.wv["a"].shape)
                    
                    for word in result:
                        try:
                            word = re.sub(reg, r'\1 \2', word)  # Hump named participles
                            for i in word.split(" "):
                                i = lemmatizer.lemmatize(i)  # Reduce the base form of the verb
                                total_vector += model.wv[i]
                        except:
                            count_word_nofind += 1
                    if count_word_nofind != 0:
                        avg_vector = total_vector / (len(result) - count_word_nofind)
                    else:  # 有些是空文件
                        avg_vector = total_vector
                    if sequence == "old":
                        old_wv = avg_vector
                    else:
                        new_wv = avg_vector
            except:
                print("nofile", os.path.join(projectname, sequence, json_file))
        with open(os.path.join(projectpath, json_file), "r+", encoding='utf8') as fp:
            json_data = json.load(fp)
            json_data['old_wv'] = old_wv.tolist()
            json_data['new_wv'] = new_wv.tolist()
            fp.seek(0)
            fp.truncate()
            fp.seek(0)
            fp.write(json.dumps(json_data))


if __name__ == '__main__':
    projects = ['activemq', 'cloudstack', 'commons-math', 'flink', 'geode', 'james-project', 'logging-log4j2', 'storm',
                'usergrid', 'zeppelin', 'biojava', 'jruby', 'jsoup']
    projects = ["AntennaPod", "deeplearning4j", "junit4", "metrics", "OpenRefine", "storio", "wiremock"]
    
    for project in tqdm(projects):
        get_project_in_json("../experiment_data/" + project, project)
