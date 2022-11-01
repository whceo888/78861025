import pandas as pd
from datasets import load_dataset,Dataset
import torch
import json
from torch.utils.data import DataLoader, ConcatDataset

# Load the dataset

df_train01 = pd.read_json("data/train.json",encoding="utf-8", orient='records')
dataset_train01 = Dataset.from_pandas(df_train01)
print(df_train01.keys())
print(len(dataset_train01))
print(dataset_train01[0])

id_id = []
for i in dataset_train01['id']:
    id_id.append(i)

id_subject = []
for i in dataset_train01['subject']:
    if i not in id_subject:
        id_subject.append(i)
print(id_subject)


flat_dataset_train01 = dataset_train01.flatten()
print(len(flat_dataset_train01))#21072
print(flat_dataset_train01[0])
#flat_dataset_train01 = flat_dataset_train01.map(flat_dataset_train01['ids']=range(0,len(flat_dataset_train01)))
#torch.save(flat_dataset_train01, "data/dataset_train01_flat.csv")



df_test01 = pd.read_json("data/test.json",encoding="utf-8", orient='records')
dataset_test01 = Dataset.from_pandas(df_test01)
print(df_test01.keys())
print(len(dataset_test01))
print(dataset_test01[0])

flat_dataset_test01 = dataset_test01.flatten()
print(len(flat_dataset_test01))#5289
print(flat_dataset_test01[0])
#torch.save(flat_dataset_test01, "data/dataset_test01_flat.csv")




#只能选一种，要么就是用pandas+Dataset读取，要么就是用load_dataset读取，不能同时用两种
#或者打开一个json文件重写


def flatten_a(example):
    example_a = {'option_list': example['option_list']['A'], 'statement': example['statement'],
                'label': example['answer'].count('A'),
                'subject': example['subject'],'type': example['type']}


    return example_a

def flatten_b(example):

    example_b = {'option_list': example['option_list']['B'], 'statement': example['statement'],
                'label': example['answer'].count('B'),
                'subject': example['subject'],'type': example['type']}

    return  example_b

def flatten_c(example):

    example_c = {'option_list': example['option_list']['C'], 'statement': example['statement'],
                'label': example['answer'].count('C'),
                'subject': example['subject'],'type': example['type']}


    return example_c

def flatten_d(example):

    example_d = {'option_list': example['option_list']['D'], 'statement': example['statement'],
                'label': example['answer'].count('D'),
                'subject': example['subject'],'type': example['type']}

    return example_d



# dataset_a = dataset_train01.map(flatten_a, remove_columns=['answer', 'id'])
# dataset_b = dataset_train01.map(flatten_b, remove_columns=['answer', 'id'])
# dataset_c = dataset_train01.map(flatten_c, remove_columns=['answer', 'id'])
# dataset_d = dataset_train01.map(flatten_d, remove_columns=['answer', 'id'])
#
# dataset_train = ConcatDataset([dataset_a, dataset_b, dataset_c, dataset_d])
# print(len(dataset_train))
# print(dataset_train[0])
#
# torch.save(dataset_train, "data/dataset_train01.csv")

# dataloader = DataLoader(dataset_a, batch_size=4)
# for i in dataloader:
#     print(i)
#     break


# 84288
# {'option_list': '未经公安机关批准回老家探亲，3日后返回', 'statement': '范某因涉嫌故意伤害罪被公安机关决定对其监视居住，范某在监视居住期间的下列哪些行为违反了法律规定的义务?', 'subject': '刑事诉讼法', 'type': 1, 'label': 1}
# {'option_list': ['未经公安机关批准回老家探亲，3日后返回', '约定合同案件的管辖法院', '构成挪用公款罪', '级别管辖'], 'statement': ['范某因涉嫌故意伤害罪被公安机关决定对其监视居住，范某在监视居住期间的下列哪些行为违反了法律规定的义务?', '根据民事诉讼法和有关司法解释，当事人可以约定下列哪些事项?', '甲为某村委会主任，利用职务之便将该村土地征用费中的50万元单独存人H银行办事处，定期2年。次日，甲伪造村委会证明，用该50万元定期存单作为质押，在H银行办事处办理个人质押贷款手续，质押贷款40万元，用于个人经营活动，无法归还。对甲的行为下列说法正确的是:', '我国民事诉讼法规定的管辖中，哪些管辖属于裁定管辖?'], 'subject': ['刑事诉讼法', None, None, '民事诉讼法'], 'type': tensor([1, 0, 1, 0]), 'label': tensor([1, 1, 1, 0])}

###test

# def flatten_testa(example):
#     example_testa = {'option_list': example['option_list']['A'], 'statement': example['statement'],
#                 'type': example['type']}
#
#
#     return example_testa
#
# def flatten_testb(example):
#
#     example_testb = {'option_list': example['option_list']['B'], 'statement': example['statement'],
#                 'type': example['type']}
#
#     return  example_testb
#
# def flatten_testc(example):
#
#     example_testc = {'option_list': example['option_list']['C'], 'statement': example['statement'],
#                 'type': example['type']}
#     return example_testc
#
# def flatten_testd(example):
#
#     example_testd = {'option_list': example['option_list']['D'], 'statement': example['statement'],
#                 'type': example['type']}
#
#     return example_testd
#
# dataset_texta = dataset_test01.map(flatten_testa, remove_columns=['id'])
# dataset_textb = dataset_test01.map(flatten_testb, remove_columns=['id'])
# dataset_textc = dataset_test01.map(flatten_testc, remove_columns=['id'])
# dataset_textd = dataset_test01.map(flatten_testd, remove_columns=['id'])
# dataset_test = ConcatDataset([dataset_texta, dataset_textb, dataset_textc, dataset_textd])
#
# print(len(dataset_test))
# print(dataset_test[0])
#
# torch.save(dataset_test, "data/dataset_test01.csv")