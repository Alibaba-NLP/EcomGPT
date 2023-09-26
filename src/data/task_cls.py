import json
import random
import os
from tqdm import tqdm
from data.utils import load_data, mapping2label
import sys
from seqeval.metrics import classification_report
from collections import Counter
import numpy as np
from rouge import Rouge
sys.path.append('../')
sys.path.append('../../')
from copy import deepcopy
from glob import glob

class Task:
    def __init__(self, dataset_folder):
        self.task_name = 'CLS'
        self.definition = 'Classification'
        self.n_examples = 20
        self.data_folder = dataset_folder
        print(dataset_folder)
        try:
            self.val_file = glob(self.data_folder + "/*test.json")[0]
        except IndexError:
            self.val_file = None
            print("no validation file founded")
        self.meta_file = glob(self.data_folder + "/*meta*.json")[0]
        self.n_per_sample = 1

        self.prompt_template = """Given the sentence {inp} \n
                            Classify the sentences.\n"""
        self.answer_template = """{out}"""

        self.val_data = None
        self.meta_data = None


    def prepare(self):
        self.val_data = load_data(self.val_file)

        print(self.meta_file)
        self.meta_data = json.load(open(self.meta_file, 'r'))


    def encode_to_input_output(self, example):
        yield dict(), dict()

    def generate(self):
        val_data = []
        for e in tqdm(self.val_data, desc='generate validation data'):
            for inp, out in self.encode_to_input_output(e):
                prompt = self.prompt_template.format(**inp)
                answer = self.answer_template.format(**out)
                val_data.append({'prompt': prompt, 'answer': answer})

        self.inst_data = {'val': val_data}

    def evaluate(self, data, results):
        pass

    def parse_answer(self, answer):
        pass



class SetScore:
    """evaluate macro and micro set p/r/f1 scores"""

    def __init__(self):
        self.n_sample = 0
        self.pred = []  # list of list
        self.true = []  # list of list

    def reset(self):  # noqa: D102
        self.n_sample = 0
        self.pred = []  # list of list
        self.true = []  # list of list

    def set_pred_true(self, pred, true):  # noqa: D102
        self.pred = pred
        self.true = true

    def update(self, batch_gold_entities, batch_pred_entities):  # noqa: D102
        self.n_sample += len(batch_gold_entities)
        self.pred.extend(batch_pred_entities)
        self.true.extend(batch_gold_entities)

    def f1(self, precision, recall):  # noqa: D102
        f1 = 0.0 if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return f1

    def result(self):  # noqa: D102
        assert len(self.pred) == len(self.true)
        M = len(self.pred)
        strict_acc = 0
        num_pred_labels = 0
        num_true_labels = 0
        num_correct_labels = 0
        total_ma_p = 0
        total_ma_r = 0
        total_ma_f1 = 0
        count = 0
        for i in range(M):
            p = set(self.pred[i])
            t = set(self.true[i])
            count += 1

            if p == t:
                strict_acc += 1

            l_p, l_t, l_intersect = len(p), len(t), len(p.intersection(t))
            num_pred_labels += l_p
            num_true_labels += l_t
            num_correct_labels += l_intersect

            if l_p == 0 or l_t == 0:
                ma_p = 0
                ma_r = 0
                ma_f1 = 0
            else:
                ma_p = l_intersect / l_p
                ma_r = l_intersect / l_t
                ma_f1 = self.f1(ma_p, ma_r)

            total_ma_p += ma_p
            total_ma_r += ma_r
            total_ma_f1 += ma_f1

        if num_pred_labels == 0 or num_true_labels == 0:
            micro_p = 0
            micro_r = 0
            micro_f1 = 0
        else:
            micro_p = num_correct_labels / num_pred_labels
            micro_r = num_correct_labels / num_true_labels
            micro_f1 = self.f1(micro_p, micro_r)

        strict_acc /= count
        macro_p = total_ma_p / count
        macro_r = total_ma_r / count
        macro_f1 = self.f1(macro_p, macro_r)
        avg_true_label = num_true_labels / M
        avg_pred_label = num_pred_labels / M

        return {
            'strict_acc': strict_acc,
            'micro_p': micro_p,
            'micro_r': micro_r,
            'micro_f1': micro_f1,
            'macro_p': macro_p,
            'macro_r': macro_r,
            'macro_f1': macro_f1,
            'avg_true_label': avg_true_label,
            'avg_pred_label': avg_pred_label,
        }





class EN_CLS(Task):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """
    def __init__(self, dataset_folder, args):
        super().__init__(dataset_folder)
        self.task_name = "EN_CLS"

        self.n_examples = 20
        self.prepare()
        self.args = args

        self.answer_template = """{out}"""
        self.definition = self.meta_data["defi"]
        self.head = "Below is an instruction that describes the {defi} task. Write a response that appropriately completes the request.\n".format(defi=self.definition)

        self.label_key, self.label_list = self.meta_data['label_key'], self.meta_data["label_set"][self.meta_data['label_key']]
        self.prompt_template = self.head + self.meta_data["test_prompt"]["instruction"]
        self.label_sep = self.meta_data["label_sep"]
        self.label_template = lambda x: self.label_sep.join(x)


        self.n_per_sample = 3
        self.max_label = 50
        self.scorer = SetScore()


    ## for eval data
    def encode_to_input_output(self, example, args):
        """
        example -> instruction
        """
        inputs = example['sentences']
        # pos_labels = [i['name'] for i in example['labels']]
        pos_labels = example['labels'][self.label_key]
        all_labels = self.label_list
        for k in range(self.n_per_sample):
            random.shuffle(all_labels)
            max_label = np.random.randint(self.max_label)
            all_labels = all_labels[:max_label]
            all_labels = list(set(all_labels + pos_labels))
            labels = self.label_template(all_labels)

            prompt = {'sentences': ''.join(inputs) if isinstance(inputs, list) else inputs,
                      'label_set': labels,
                      'ex': ''}
            answer = {'out': self.label_sep.join(pos_labels)}

            all_prompt = [self.prompt_template.format(**prompt)]
            answer = self.answer_template.format(**answer)

            yield all_prompt, answer

            if len(pos_labels) == 1:
                break


    def parse_answer(self, answer):
        if len(answer) == 0:
            return("error")
        return answer.strip().replace(".", '')

    def parse_answer_to_list(self, answer):
        if len(answer) == 0:
            return("error")
        return answer.split(self.label_sep)

    def parse_gold_to_list(self, gold):
        if len(gold) == 0:
            return("error")
        return gold.split(self.label_sep)

    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            # print(d)
            # print(pred_result)
            gold_answer = d['answer']

            gold_answers.append(gold_answer)
            all_pred.append(self.parse_answer(pred_result))
        scores = rouge.get_scores(all_pred, gold_answers, avg=True)
        self.scorer.update([self.parse_gold_to_list(i) for i in gold_answers], [self.parse_answer_to_list(j) for j in all_pred])
        res = self.scorer.result()
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  res['micro_f1'],
                      'macro-f1':  res['macro_f1'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'setscore': res, 'rouge': scores}


        return detailed_report, key_scores