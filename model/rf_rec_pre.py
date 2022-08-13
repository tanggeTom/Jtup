import json
import os
from numpy import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from sklearn.model_selection import train_test_split


# read json file and get features
def read_json(filename):
    global positive_num, negative_num
    with open(filename, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
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
        pre_cyclomatic_complexity = str(pre_cyclomatic_complexity).replace(",", "")  # 存在1,234过千的数字
        last_cyclomatic_complexity = str(last_cyclomatic_complexity).replace(",", "")
        cyclomatic_complexity_diff = int(last_cyclomatic_complexity) - int(pre_cyclomatic_complexity)
        pre_entropy = 0 if not json_data.get('pre_entropy') else json_data["pre_entropy"]
        last_entropy = 0 if not json_data.get('last_entropy') else json_data["last_entropy"]
        new_wv = json_data['new_wv']
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
        feature = [add_annotation_line, add_call_line, add_classname_line, add_condition_line, add_field_line,
                   add_import_line, add_packageid_line, add_parameter_line, add_return_line, del_annotation_line,
                   del_call_line, del_classname_line, del_condition_line, del_field_line, del_import_line,
                   del_packageid_line, del_parameter_line, del_return_line, insert_num, update_num, move_num,
                   delete_num, clusters_num, actions_num, pre_cyclomatic_complexity, last_cyclomatic_complexity,
                   cyclomatic_complexity_diff, pre_entropy, last_entropy]
        feature.extend(new_wv)
        features.append(feature)
        if json_data['sample_type'] == "POSITIVE":
            positive_num += 1
            target.append(1)
        else:
            negative_num += 1
            target.append(0)
        return features



features = []
target = []
positive_num = 0
negative_num = 0
if __name__ == '__main__':
    
    projects = ['activemq', 'cloudstack', 'commons-math', 'flink', 'geode', 'james-project', 'logging-log4j2', 'storm',
                'usergrid', 'zeppelin', 'pmd', 'biojava', 'jruby', 'jsoup',
                "AntennaPod", "deeplearning4j", "junit4", "metrics", "OpenRefine", "storio", "wiremock"]
    for dir in projects:
        for file in os.listdir('../experiment_data/' + dir):
            read_json('../experiment_data/' + dir + '/' + file)
    # Print the number of positive and negative samples
    print('positive', positive_num)
    print('negative', negative_num)
    # test and training sets 0.1-0.9
    features_np = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(features_np, target, test_size=0.1)
    rgs = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto')
    rgs = rgs.fit(x_train, y_train)
    predict = rgs.predict(x_test)
    
    print('Accuracy score:', accuracy_score(y_test, predict))
    print('Recall:', recall_score(y_test, predict))
    print('F1-score:', f1_score(y_test, predict))
    print('Precision score:', precision_score(y_test, predict))
