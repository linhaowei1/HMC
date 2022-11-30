import os
from networks.deberta import DebertaV2ForHMC
from transformers import AutoTokenizer
class Relation:

    def __init__(self, file_path='tools/case_classification.txt'):

        relation = {}
        self.tree = {}
        with open(file_path, 'r') as f:
            lst = f.readlines()
        one, two = None, None
        for line in lst:
            name, idx = line.split()
            if int(idx) == 1:
                one = name
                relation[one] = {"未知":["未知"]}
            elif int(idx) == 2:
                two = name
                relation[one][two] = ["未知"]
                self.tree[name] = one
            elif int(idx) == 3:
                relation[one][two].append(name)
                self.tree[name] = two
        self.relation = relation
        self._prepare_all_label()

        
    def prepare_label(self, idx, name):
        first_class_name, second_class_name, third_class_name = "未知","未知","未知"

        if idx == 1 or idx == -1:
            first_class_name = name    
        elif idx == 2:
            first_class_name = self.tree[name]
            second_class_name = name
        else:
            first_class_name = self.tree[self.tree[name]]
            second_class_name = self.tree[name]
            third_class_name = name
        return {
            'first': first_class_name,
            'second': "{}-{}".format(first_class_name, second_class_name),
            'third':"{}-{}-{}".format(first_class_name, second_class_name, third_class_name)
        }
    
    def _prepare_all_label(self):
        first_class_labels = {key:i for i,key in enumerate(self.relation.keys())}
        second_class_labels = {}
        third_class_labels = {}
        i = 0
        j = 0
        for key in self.relation.keys():
            for name in self.relation[key].keys():
                second_class_labels[key+'-'+name] = i
                i += 1
                for value in self.relation[key][name]:
                    third_class_labels[key+'-'+name+'-'+value] = j
                    j += 1
        self.idx2label = {
            'first': list(first_class_labels.keys()),
            'second': list(second_class_labels.keys()),
            'third': list(third_class_labels.keys())
        }
        self.label2idx = {
            'first': {name:idx for idx,name in enumerate(first_class_labels)},
            'second': {name:idx for idx,name in enumerate(second_class_labels)},
            'third': {name:idx for idx,name in enumerate(third_class_labels)}
        }

def lookfor_model(args):
    model = DebertaV2ForHMC.from_pretrained(args.model_name_or_path, args=args)
    tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese')
    label_func = lambda label: tokenizer(
        label,
        padding="max_length", 
        truncation=True, 
        max_length=args.max_length, 
        return_tensors="pt"
    )
    first = label_func(args.relation.idx2label['first'])
    second = label_func(args.relation.idx2label['second'])
    third = label_func(args.relation.idx2label['third'])
    model.register_buffer('first_input_ids', first['input_ids'])
    model.register_buffer('second_input_ids', second['input_ids'])
    model.register_buffer('third_input_ids', third['input_ids'])
    model.register_buffer('first_attention_mask', first['attention_mask'])
    model.register_buffer('second_attention_mask', second['attention_mask'])
    model.register_buffer('third_attention_mask', third['attention_mask'])

    return model, tokenizer

if __name__ == '__main__':
    r = Relation()
