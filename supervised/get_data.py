import pandas as pd
import random
import json,os,re
import numpy as np
from tqdm import tqdm

def data_process1(data_dir,out_dir):
    if not os.path.exists(data_dir):
        os.makedirs(in_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    path = os.path.join(data_dir, 'train_data.txt')
    with open(path, 'r', encoding='utf-8') as fp:
        data = fp.readlines()
    all_data = []
    for sample in data:
        all_data.append(json.loads(sample))

    support_list = []
    query_list = []
    for sample in all_data:
        category_name = sample['category_name']
        category_description = sample['category_description'].replace('\t', '##')
        support_set = sample['support_set']
        query_set = sample['query_set']
        for support in support_set:
            text = support.get('text').replace('\t', '##')
            support_list.append(f"{category_name}：{category_description}\t{text}\t{support.get('label')}")

        len_add = int(0.7 * len(query_set))
        for query in query_set[:len_add]:
            text = query.get('text').replace('\t', '##')
            support_list.append(f"{category_name}：{category_description}\t{text}\t{query.get('label')}")

        for query in query_set[len_add:]:
            text = query.get('text').replace('\t', '##')
            query_list.append(f"{category_name}：{category_description}\t{text}\t{query.get('label')}")

    path = os.path.join(data_dir, 'test_data.txt')
    with open(path, 'r', encoding='utf-8') as fp:
        data = fp.readlines()
    all_data = []
    for sample in data:
        all_data.append(json.loads(sample))

    task_id_list = []
    record_id_list = []
    category_description_list = []
    text_list = []

    for sample in all_data:
        task_id = sample['task_id']
        category_name = sample['category_name']
        category_description = sample['category_description'].replace('\t', '##')
        support_set = sample['support_set']
        query_set = sample['query_set']
        for support in support_set[:4]:
            text = support.get('text').replace('\t', '##')
            support_list.append(f"{category_name}：{category_description}\t{text}\t{support.get('label')}")

        for support in support_set[4:]:
            text = support.get('text').replace('\t', '##')
            query_list.append(f"{category_name}：{category_description}\t{text}\t{support.get('label')}")
        # 进行负采样
        false_sample_list = [f"{category_name}：{category_description}\t" + '\t'.join(sample.split('\t')[1:])[:-1] + '0'
                             for sample in random.sample(support_list, 1000) if sample.split('：')[0] != category_name][
                            :80]
        if len(false_sample_list) < 80:
            s = 1 / 0
        support_list += false_sample_list[:65]
        query_list += false_sample_list[65:]

        # 测试集
        for query in query_set:
            text = query.get('text').replace('\t', '##')
            record_id = query.get('record_id')
            task_id_list.append(task_id)
            record_id_list.append(record_id)
            category_description_list.append(f"{category_name}：{category_description}")
            text_list.append(text)

    print("补充后................................")
    print('len_support_list：', len(support_list))
    print('len_query_list：', len(query_list))
    print("测试集比例：", len(query_list) / (len(query_list) + len(support_list)))

    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as fp:
        for sample in support_list:
            fp.write(sample.strip() + '\n')

    with open(os.path.join(out_dir, 'dev.txt'), 'w', encoding='utf-8') as fp:
        for sample in query_list:
            fp.write(sample.strip() + '\n')

    df = pd.DataFrame({'task_id': task_id_list, 'record_id': record_id_list, 'description': category_description_list,
                       'text': text_list})
    df.to_csv(os.path.join(out_dir,'test.csv'), index=0, encoding='utf-8-sig')
    print("Done................")


def data_process2(in_dir,out_dir,file):
    '''将数据进行切分'''
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    path = os.path.join(in_dir, file)
    with open(path, 'r', encoding='utf-8') as fp:
        data = fp.readlines()

    all_data = []
    for sample in tqdm(data):
        describe, text, label = sample.strip().split('\t')
        all_data.extend([f"{describe}\t{s}\t{label}" for s in re.split(r'【[^】]+】：', text) if len(s)>=3])

    all_data = set(all_data)
    with open(os.path.join(out_dir, file),'w',encoding='utf-8') as fp:
        for sample in all_data:
            fp.write(sample + '\n')

    print("Done................")
    return len(all_data)


if __name__ == "__main__":
    in_dir = './data/comment_classify2/'
    out_dir = './data/comment_classify3/'
    data_process1(in_dir,out_dir)

    file = 'train.txt'
    support_len = data_process2(out_dir,out_dir,file)

    file = 'dev.txt'
    query_len = data_process2(out_dir,out_dir,file)

    print('len_support_list：', support_len)
    print('len_query_list：', query_len)
    print("测试集比例：", query_len / (query_len + support_len))

