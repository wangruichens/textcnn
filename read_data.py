# @Time    : 18-11-5
# @Author  : wangrc
# @Refers   : 
# @Outputs   : 
# @Desc   : 
import tensorflow as tf
import pandas as pd
import jieba
import re
import os


def sp_title(title):
    rule = '[^/.0-9a-zA-Z\u4e00-\u9fa5]+'
    c = re.sub(rule, '', title)
    sp = jieba.lcut(c)
    if len(sp) < 3:
        return ''
    res = ''
    for s in sp:
        res += str(s) + ' '
    return res + '\n'


def write_append_data(path, l, mode='w'):
    with open(path, mode) as f:
        for item in l:
            f.write(item)


def load_nlpcc(path):
    train_x = []
    train_y_cat = []
    with open(path, 'r') as f:
        content = f.readlines()
        for c in content:
            train_y_cat.append(c.split('\t')[0])
            train_x.append(c.split('\t')[1])
    # train_y = pd.get_dummies(train_y_cat).values.tolist()
    return train_x, train_y_cat


def format_nlpcc_data():
    x1, y1 = load_nlpcc('/home/wangrc/data/nlpcc_data/word/train.txt')
    x2, y2 = load_nlpcc('/home/wangrc/data/nlpcc_data/word/dev.txt')
    print(len(x1), len(x2))
    x = x1 + x2
    y = y1 + y2
    print(len(x))
    # 8 categories except xiaohua
    entertainment = []
    regimen = []
    fashion = []
    finance = []
    society = []
    military = []
    car = []
    sports = []
    for i in range(len(y)):
        if y[i] == 'entertainment':
            entertainment.append(x[i])
        if y[i] == 'regimen':
            regimen.append(x[i])
        if y[i] == 'fashion':
            fashion.append(x[i])
        if y[i] == 'finance':
            finance.append(x[i])
        if y[i] == 'society':
            society.append(x[i])
        if y[i] == 'military':
            military.append(x[i])
        if y[i] == 'car':
            car.append(x[i])
        if y[i] == 'sports':
            sports.append(x[i])
    write_append_data('entertainment', entertainment)
    write_append_data('regimen', regimen)
    write_append_data('fashion', fashion)
    write_append_data('finance', finance)
    write_append_data('society', society)
    write_append_data('military', military)
    write_append_data('car', car)
    write_append_data('sports', sports)


def format_app_data():
    y_name = ['xiaohua', 'yule', 'jiankang', 'shishang', 'caijing', 'shehui', 'junshi', 'qiche', 'tiyu', 'toutiao']
    path_app = '/home/wangrc/data/app_data/app_self/'

    for n in y_name:
        train_x = []
        content = pd.read_csv(path_app + n, header=None)[0].tolist()
        for c in content:
            res = sp_title(c)
            if res != '':
                train_x.append(res)
        write_append_data(n, train_x)


def format_zj_data():
    y_name = ['xiaohua', 'yule', 'jiankang', 'shishang', 'caijing', 'shehui', 'junshi', 'qiche', 'tiyu', 'xingzuo']
    path_app = '/home/wangrc/data/app_data/zhenjun/'

    for n in y_name:
        train_x = []
        if os.path.isfile(path_app + n):
            with open(path_app + n, 'r') as f:
                content = f.readlines()
                for c in content:
                    res = sp_title(c)
                    if res != '':
                        train_x.append(res)
            print(n, len(train_x))
            write_append_data(n, train_x)


def agg_all_data():
    path_nlpcc = '/home/wangrc/data/app_data/nlpcc_format/'
    path_zj = '/home/wangrc/data/app_data/zj_format/'
    path_app = '/home/wangrc/data/app_data/app_format/'
    path_all = [path_nlpcc, path_zj, path_app]
    y_name = ['xiaohua', 'yule', 'jiankang', 'shishang', 'caijing', 'shehui', 'junshi', 'qiche', 'tiyu','xingzuo']
    for n in y_name:
        train_x = []
        for p in path_all:
            if os.path.isfile(p + n):
                with open(p + n, 'r') as f:
                    content = f.readlines()
                    for c in content:
                        train_x.append(c)
        print(n, len(train_x))
        write_append_data(n, train_x)


def load_all_data():
    path = '/home/wangrc/data/app_data/all_format/'
    y_name = ['xiaohua', 'yule', 'jiankang', 'shishang', 'caijing', 'shehui', 'junshi', 'qiche', 'tiyu','xingzuo']
    cat_total = 30000
    train_x = []
    train_y = []
    for n in y_name:
        cat_count = 0
        if os.path.isfile(path + n):
            with open(path + n, 'r') as f:
                content = f.readlines()
                for c in content:
                    cat_count += 1
                    if cat_count <= cat_total:
                        train_y.append(n)
                        train_x.append(c)

    # Report the count
    for n in y_name:
        print(n, train_y.count(n))
    # df=pd.get_dummies(train_y)
    # print(df.head(5))
    train_y = pd.get_dummies(train_y).values.tolist()
    return train_x, train_y


def load_toutiao():
    train_x = []
    with open('/home/wangrc/data/app_data/app_eval/toutiao', 'r') as f:
        content = f.readlines()
        for c in content:
            train_x.append(c)
    return train_x, None