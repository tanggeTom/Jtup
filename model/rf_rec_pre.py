import json
import os
from operator import attrgetter

import psutil
from imblearn.under_sampling import RandomUnderSampler
from numpy import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import time


class Commit(object):

    def __init__(self, file_name, time, project, feature, target):
        self.file_name = file_name
        self.time = time
        self.project = projects
        self.feature = feature
        self.target = target

    def __str__(self):
        return "(" + self.file_name + ", " + str(self.time) + ")"



# read json file and get features
def read_json(filename, project):
    global positive_num, negative_num
    with open(filename, 'r', encoding='utf8') as fp:
        try:
            json_data = json.load(fp)
        except:
            print(filename)
        add_annotation_line = json_data['add_annotation_line']
        add_call_line = json_data['add_call_line']
        add_classname_line = json_data['add_classname_line']
        add_condition_line = json_data['add_condition_line']
        add_field_line = json_data['add_field_line']
        add_import_line = json_data['add_import_line']
        add_packageid_line = json_data['add_packageid_line']
        add_parameter_line = json_data['add_parameter_line']
        add_return_line = json_data['add_return_line']
        del_annotation_line = json_data['del_annotation_line']
        del_call_line = json_data['del_call_line']
        del_classname_line = json_data['del_classname_line']
        del_condition_line = json_data['del_condition_line']
        del_field_line = json_data['del_field_line']
        del_import_line = json_data['del_import_line']
        del_packageid_line = json_data['del_packageid_line']
        del_parameter_line = json_data['del_parameter_line']
        del_return_line = json_data['del_return_line']
        pre_cyclomatic_complexity = 0 if not json_data.get('pre_cyclomatic_complexity') else json_data[
            "pre_cyclomatic_complexity"]
        last_cyclomatic_complexity = 0 if not json_data.get('last_cyclomatic_complexity') else json_data[
            "last_cyclomatic_complexity"]
        pre_cyclomatic_complexity = int(str(pre_cyclomatic_complexity).replace(",", ""))  # 存在1,234过千的数字
        last_cyclomatic_complexity = int(str(last_cyclomatic_complexity).replace(",", ""))
        cyclomatic_complexity_diff = int(last_cyclomatic_complexity) - int(pre_cyclomatic_complexity)
        pre_entropy = 0 if not json_data.get('pre_entropy') else float(json_data["pre_entropy"])
        last_entropy = 0 if not json_data.get('last_entropy') else float(json_data["last_entropy"])
        new_wv = json_data['new_wv']
        new_wv_2 = json_data['new_wv_2']
        new_wv_7 = json_data['new_wv_7']
        insert_num = 0 if not json_data.get('insert') else json_data["insert"]
        update_num = 0 if not json_data.get('update') else json_data["update"]
        move_num = 0 if not json_data.get('move') else json_data["move"]
        delete_num = 0 if not json_data.get('delete') else json_data["delete"]
        clusters_num = 0
        if insert_num != 0:
            clusters_num += 1
        if update_num != 0:
            clusters_num += 1
        if move_num != 0:
            clusters_num += 1
        if delete_num != 0:
            clusters_num += 1
        actions_num = delete_num + move_num + update_num + insert_num
        modified_code_lines_level = json_data["modified_code_lines_level"]
        normalized_modified_code_lines = json_data["normalized_modified_code_lines"]
        non_normalized_modified_code_lines = json_data["normalized_modified_code_lines"]

        modified_method_number_level = json_data["modified_method_number_level"]
        pro_modified_method_number_inclass = json_data["pro_modified_method_number_inclass"]

        feature = [
            add_annotation_line, add_call_line, add_classname_line, add_condition_line, add_field_line,
                   add_import_line, add_packageid_line, add_parameter_line, add_return_line, del_annotation_line,
                   del_call_line, del_classname_line, del_condition_line, del_field_line, del_import_line,
                   del_packageid_line, del_parameter_line, del_return_line, insert_num, update_num, move_num,
                   delete_num, clusters_num, actions_num,
                   pre_cyclomatic_complexity, last_cyclomatic_complexity,
                   cyclomatic_complexity_diff,
            pre_entropy, last_entropy,
                   modified_code_lines_level,
                   normalized_modified_code_lines, non_normalized_modified_code_lines, modified_method_number_level,
                   pro_modified_method_number_inclass
        ]
        feature.extend(new_wv)
        features.append(feature)
        try:
            prod_time = json_data["prod_time"]["$date"]
        except:
            prod_time = json_data["prod_time"]
            prod_time = datetime.datetime.strptime(prod_time, "%a %b %d %H:%M:%S CST %Y")
            prod_time = (time.mktime(prod_time.timetuple()) * 1000)
        commit1 = Commit()
        commit1.time = prod_time
        commit1.file_name = filename
        commit1.project = project
        commit1.feature = feature
        if json_data['sample_type'] == "POSITIVE":
            positive_num += 1
            commit1.target = 1
            target.append(1)
        else:
            negative_num += 1
            commit1.target = 0
            target.append(0)
        commit_list.append(commit1)
        return features


commit_list = []
features = []
target = []
features_test = []
target_test = []
positive_num = 0
negative_num = 0
project_dict = dict()
project_train_dict = dict()
if __name__ == '__main__':
    start_time = time.time()
    projects = ['activemq', 'cloudstack', 'commons-math', 'flink', 'geode', 'james-project', 'logging-log4j2', 'storm',
                'usergrid', 'zeppelin', 'biojava', 'jruby', 'jsoup',
                "AntennaPod", "deeplearning4j", "junit4", "metrics", "OpenRefine", "storio", "wiremock"]
    for dir in projects:
        for file in os.listdir('../experiment_data/' + dir):
            read_json('../experiment_data/' + dir + '/' + file, dir)
    features = []
    target = []
    commit_list.sort(key=attrgetter('time'))
    for commit in commit_list:
        project_dict[commit.project] = project_dict.get(commit.project, 0) + 1
    print(project_dict)
    for commit in commit_list:
        if project_dict.get(commit.project) * 0.9 < project_train_dict.get(commit.project, 0):  # 超过了规定的比例
            features_test.append(commit.feature)
            target_test.append(commit.target)
        else:
            features.append(commit.feature)
            target.append(commit.target)
            project_train_dict[commit.project] = project_train_dict.get(commit.project, 0) + 1

    rgs = RandomForestClassifier()
    print(features[0],target[0])
    rgs = rgs.fit(features, target)
    predict = rgs.predict(features_test)
    print(len(predict))
    print('Accuracy score:', accuracy_score(target_test, predict))
    print('Recall:', recall_score(target_test, predict))
    print('F1-score:', f1_score(target_test, predict))
    print('Precision score:', precision_score(target_test, predict))
    end_time = time.time()
    runtime = end_time - start_time
    print("running time:", runtime, "seconds")

    process = psutil.Process()
    memory_usage = process.memory_info().rss
    print("memory usage:", memory_usage, "bytes")
