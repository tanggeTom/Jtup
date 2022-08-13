import re
import os
from tqdm import tqdm
import json


# The output is a collection of CMD directives using the tool Gumtree
def gumtree_out_cmd(project_path, project_name):
    total_cmds = ""
    for i in tqdm(os.listdir("../experiment_data//" + project_name)):
        with open(os.path.join("../experiment_data", project_name, i), "r") as f:
            json_data = json.load(f)
            prod_sha1 = json_data["prod_sha1"]
            prod_path = json_data["prod_path"]
            file_v1 = os.path.join(project_path, str(prod_sha1), "old", prod_path)
            file_v2 = os.path.join(project_path, str(prod_sha1), "new", prod_path)
            cmd_str = ""
            blank_class_path = "" # An empty Java file path
            if not os.path.exists(file_v1):  # lack old version
                cmd_str = ".\gumtree textdiff '{}' '{}' > {}".format(file_v2, blank_class_path,
                                                                     os.path.join(project_name + "_git", "res",
                                                                                  i.replace(".json", ".txt")))
            elif not os.path.exists(file_v2):  # lack new version
                cmd_str = ".\gumtree textdiff '{}' '{}' > {}".format(blank_class_path, file_v1,
                                                                     os.path.join(project_name + "_git", "res",
                                                                                  i.replace(".json", ".txt")))
            elif os.path.exists(file_v1) and os.path.exists(file_v2):
                cmd_str = ".\gumtree textdiff '{}' '{}' > {}".format(file_v2, file_v1,
                                                                     os.path.join(project_name + "_git", "res",
                                                                                  i.replace(".json", ".txt")))
            total_cmds += cmd_str + "\n"
    return total_cmds


def write_json(project_name):
    path = ""
    for i in tqdm(os.listdir("../experiment_data//" + project_name)):
        with open(os.path.join(path, i.replace(".json", ".txt")), "r", encoding="utf16") as fp:
            content = fp.read()
            insert_num = content.count("insert")
            update_num = content.count("update")
            move_num = content.count("move")
            delete_num = content.count("delete")
            # Read the Gumtree file
            with open('..//experiment_data//' + project_name + '//' + i, "r+",
                      encoding='utf8') as fp:
                # print(fp.name)
                json_data = json.load(fp)
                json_data['insert'] = insert_num
                json_data['update'] = update_num
                json_data['move'] = move_num
                json_data['delete'] = delete_num
                fp.seek(0)
                fp.truncate()
                fp.seek(0)
                fp.write(json.dumps(json_data))
