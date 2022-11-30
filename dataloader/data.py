import json
import os
from datasets import Dataset, DatasetDict
def get_dataset(args):
    with open('/data1/haowei/haowei/dl/train_data_20220406_10w.txt','r') as f:
        data = [eval(x) for x in f.readlines()]
    text = [x['aq'] for x in data]
    labels = [args.relation.prepare_label(x['aydj'], x['ay']) for x in data]
    first_label = [x['first'] for x in labels]
    second_label = [x['second'] for x in labels]
    third_label = [x['third'] for x in labels]
    dataset = Dataset.from_dict(
        {
            'text': text,
            'first_label': first_label,
            'second_label': second_label,
            'third_label': third_label
        }
    )
    dataset = dataset.train_test_split(test_size=0.0005)
    with open('/home/haowei/haowei/NLP/tools/test_data_2022_1w.txt','r') as f:
        data = [eval(x) for x in f.readlines()]
    text = [x['aq'] for x in data]
    return DatasetDict(
        {
            'train': dataset['train'],
            'test': dataset['test'],
            'pred': Dataset.from_dict({'text': text})
        }
    )

