import os
import subprocess

order = 3
fold_num = 10
options = '-ENTROPY -BACKOFF -TEST -FILES'
options_cache = '-ENTROPY -BACKOFF -TEST -CACHE -CACHE_ORDER 3 -CACHE_DYNAMIC_LAMBDA -FILE_CACHE -FILES'
options_window5000 = '-ENTROPY -BACKOFF -TEST -CACHE -CACHE_ORDER 3 -CACHE_DYNAMIC_LAMBDA -WINDOW_CACHE -WINDOW_SIZE 5000 -FILES'
train_file = './data/sample_project/project_output/fold0.train'
test_file = './data/sample_project/newold_output/storiotest'
scope_file = './data/sample_project/project_output/fold0.scope'
cp = subprocess.run(
    './code/completion {} -NGRAM_FILE {}.{}grams -NGRAM_ORDER {} -SCOPE_FILE {} -INPUT_FILE {} -OUTPUT_FILE {}.output'.format
    (options, train_file, order, order, scope_file, test_file, test_file), shell=True, stdout=subprocess.PIPE)
if cp.returncode:
    raise (IOError, './code/completion指令执行失败')
lines = cp.stdout.decode().split('\n')
for line in lines:
    if 'Total tokens:' in line:
        print(line.strip('\n'))
    if 'Entropy:' in line:
        print(line)
