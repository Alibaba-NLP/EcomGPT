import json
import random
import os
from glob import glob
from tqdm import tqdm
import sys
sys.path.append('../')
sys.path.append('../../')
from data.utils import load_data, mapping2label
from seqeval.metrics import classification_report
from collections import Counter
import numpy as np
from rouge import Rouge
from copy import deepcopy

class ExtScore:
    def evaluate(self, gold, pred):
        report = classification_report(gold, pred, output_dict=True)
        return report

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


class Task:
    def __init__(self, sub_folder):
        self.task_name = 'NER'
        self.definition = 'span extraction'
        self.n_examples = 20
        self.data_folder = sub_folder
        self.val_file = glob(self.data_folder + "/*test.json")[0]
        self.meta_file = glob(self.data_folder + "/*meta*.json")[0]
        self.n_per_sample = 1

        self.val_data = None
        self.example_data = None
        self.inst_data = None
        
        self.scorer = SetScore()
        self.ext_scorer = ExtScore()

    def prepare(self):
        self.val_data = load_data(self.val_file)

        print(self.meta_file)
        self.meta_data = json.load(open(self.meta_file, 'r'))

    def encode_to_input_output(self, example):
        yield dict(), dict()


    def sample(self, data, n=5, max_per_example=1):
        sample_data = []
        for e in tqdm(data[:n], desc='sample data'):
            cnt = 0
            for inp, out in self.encode_to_input_output(e):
                if cnt >= max_per_example:
                    break
                prompt = self.prompt_template.format(**inp)
                answer = self.answer_template.format(**out)
                sample_data.append({'prompt': prompt, 'answer': answer, 'example': e})
                cnt += 1
        return sample_data

    def get_metrics(self, y_true, y_pred):
        total_pred = 0
        total_gt = 0
        tp = 0
        for gt, pred in zip(y_true, y_pred):
            gt_list = gt.split(', ')
            pred_list = pred.split(', ')
            total_pred+=len(pred_list)
            total_gt+=len(gt_list)
            for gt_val in gt_list:
                for pred_val in pred_list:
                    if pred_val.lower() in gt_val.lower() or gt_val.lower() in pred_val.lower():
                        tp+=1
                        break
        p = tp/total_pred
        r = tp/total_gt
        return p, r, 2*p*r/(p+r), None

    def evaluate(self, data, results):
        pass

    def parse_answer(self, answer):
        pass


class EN_GEN(Task):
    def __init__(self, sub_folder, args):
        super(EN_GEN, self).__init__(sub_folder)

        self.task_name = 'EN_GEN'
        self.prepare()
        self.definition = self.meta_data["defi"]
        self.n_examples = 20
        self.args = args
        
        self.head = "Below is an instruction that describes the {defi} task. Write a response that appropriately completes the request.\n".format(defi=self.definition)
        self.answer_template = """{out}"""
        self.prompt_template = self.head + self.meta_data["test_prompt"]["instruction"]
        self.label_sep = self.meta_data["label_sep"]
        self.output_template = lambda x: self.label_sep.join(x)

        print(self.task_name)

    def encode_to_input_output(self, example, args):
        """
        example -> instruction
        """
        inputs = example['sentences']
        pos_labels = [i["term"] for i in example['spans']]
        for k in range(self.n_per_sample):
            prompt = {'sentences': ''.join(inputs) if isinstance(inputs, list) else inputs,
                    'label_set': '',
                      'ex': ''}
            answer = {'out': self.label_sep.join(pos_labels)}

            all_prompt = [self.prompt_template.format(**prompt)]
            answer = self.answer_template.format(**answer)
            
            yield all_prompt, answer

            if len(pos_labels) == 1:
                break

    def save_data(self):
        for k, v in self.inst_data.items():
            save_path = os.path.join(self.data_folder,
                                     '{split}.{task}.inst.json'.format(split=k, task=self.task_name))
            with open(save_path, 'w') as f:
                dumped = json.dumps(v, indent=4 ,ensure_ascii=False)
                f.write(dumped)
            print("save data to: {}".format(save_path))


    def parse_answer(self, answer):
        if len(answer) == 0:
            print("error")
        return answer.strip()

    def parse_glod(self, gold_answer):
        return gold_answer.strip()

    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']
            gold_answers.append(gold_answer)
            if len(pred_result)==0:
                pred_result="error"
            pred_answers.append(pred_result)

            all_pred.append(self.parse_answer(pred_result))
            all_gold.append(self.parse_glod(gold_answer))
        # report = self.get_metrics(all_pred, all_gold)
        scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
        self.scorer.update(all_gold, all_pred)
        res = self.scorer.result()
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  res['micro_f1'],
                      'macro-f1':  res['macro_f1'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': res, 'rouge': scores}

        return detailed_report, key_scores

class EN_EXTRACT(Task):
    def __init__(self, sub_folder, args):
        super(EN_EXTRACT, self).__init__(sub_folder)

        self.task_name = 'EN_EXTRACT'
        self.prepare()
        self.definition = self.meta_data["defi"]
        self.n_examples = 20
        self.args = args
        self.head = "Below is an instruction that describes the {defi} task. Write a response that appropriately completes the request.\n".format(defi=self.definition)
        self.answer_template = """{out}"""
        self.prompt_template = self.head + self.meta_data["test_prompt"]["instruction"]
        self.label_sep = self.meta_data["label_sep"]
        self.output_template = lambda x: self.label_sep.join(x)

        print(self.task_name)

    def encode_to_input_output(self, example, args):
        """
        example -> instruction
        """
        inputs = example['sentences']
        pos_labels = [i["term"] for i in example['spans']]
        for k in range(self.n_per_sample):
            prompt = {'sentences': ''.join(inputs) if isinstance(inputs, list) else inputs,
                    'label_set': '',
                      'ex': ''}
            answer = {'out': self.label_sep.join(pos_labels)}

            all_prompt = [self.prompt_template.format(**prompt)]
            answer = self.answer_template.format(**answer)
            
            yield all_prompt, answer

            if len(pos_labels) == 1:
                break


    def save_data(self):
        for k, v in self.inst_data.items():
            save_path = os.path.join(self.data_folder,
                                     '{split}.{task}.inst.json'.format(split=k, task=self.task_name))
            with open(save_path, 'w') as f:
                dumped = json.dumps(v, indent=4 ,ensure_ascii=False)
                f.write(dumped)
            print("save data to: {}".format(save_path))


    def parse_answer(self, answer):
        if len(answer) == 0:
            print("error")
        return answer.strip().split(self.label_sep)

    def parse_glod(self, gold_answer):
        return gold_answer.strip().split(self.label_sep)

    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']
            if len(pred_result.replace(".","").strip())==0:
                pred_result = "error"
            gold_answers.append(gold_answer)
            pred_answers.append(pred_result)
            all_pred.append(self.parse_answer(pred_result))
            all_gold.append(self.parse_glod(gold_answer))
        # report = self.get_metrics(all_pred, all_gold)
        scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
        self.scorer.update(all_gold, all_pred)
        res = self.scorer.result()
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  res['micro_f1'],
                      'macro-f1':  res['macro_f1'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': res, 'rouge': scores}

        return detailed_report, key_scores

class EN_CONDITION_EXTRACT(Task):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """
    def __init__(self, sub_folder, args):
        super(EN_CONDITION_EXTRACT, self).__init__(sub_folder)

        self.task_name = 'EN_CONDITION_EXTRACT'
        self.prepare()
        self.definition = self.meta_data["defi"]
        self.n_examples = 20
        self.args = args

        self.head = "Below is an instruction that describes the {defi} task. Write a response that appropriately completes the request.\n".format(defi=self.definition)
        self.answer_template = """{out}"""
        self.meta_data["test_prompt"]["instruction"]
        self.prompt_template = self.head + self.meta_data["test_prompt"]["instruction"]

        
        self.label_sep = self.meta_data["label_sep"]
        self.output_template = lambda x: self.label_sep.join(x)
        self.label_key, self.label_list = self.meta_data['label_key'], self.meta_data["label_set"][self.meta_data['label_key']]

        print(self.task_name)


    def encode_to_input_output(self, example, args):
        """
        example -> instruction
        """
        inputs = example['sentences']
        pos_labels = [i["term"] for i in example['spans']]
        for k in range(self.n_per_sample):
            prompt = {'sentences': ''.join(inputs) if isinstance(inputs, list) else inputs,
                    'label_set': self.label_sep.join(self.meta_data["label_set"][self.meta_data["label_key"]]),
                      'ex': ''}
            answer = {'out': ', '.join(pos_labels)}
            all_prompt = [self.prompt_template.format(**prompt)]
            answer = self.answer_template.format(**answer)
            yield all_prompt, answer

            if len(pos_labels) == 1:
                break


    def save_data(self):
        for k, v in self.inst_data.items():
            save_path = os.path.join(self.data_folder,
                                     '{split}.{task}.inst.json'.format(split=k, task=self.task_name))
            with open(save_path, 'w') as f:
                dumped = json.dumps(v, indent=4 ,ensure_ascii=False)
                f.write(dumped)
            print("save data to: {}".format(save_path))


    def parse_answer(self, answer):
        if len(answer) == 0:
            print("error")
        return answer.split(self.label_sep)

    def parse_glod(self, gold_answer):
        return gold_answer.split(self.label_sep)

    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']
            if len(pred_result.strip())<=1:
                pred_result="error"
            if len(gold_answer)==0:
                gold_answer="error"
            gold_answers.append(gold_answer)
            pred_answers.append(pred_result)

            all_pred.append(self.parse_answer(pred_result))
            all_gold.append(self.parse_glod(gold_answer))
        # report = self.get_metrics(all_pred, all_gold)
        for pred_answer, gold_answer in zip(pred_answers, gold_answers):
            try:
                rouge.get_scores([pred_answer], [gold_answer], avg=True)
            except:
                print("empty!!")
                print("1",pred_answer)
                print("2",gold_answer)
                exit()
        scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
        self.scorer.update(all_gold, all_pred)
        res = self.scorer.result()
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  res['micro_f1'],
                      'macro-f1':  res['macro_f1'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': res, 'rouge': scores}

        return detailed_report, key_scores

class EN_NER(Task):
    """
    1. 原始数据 -》 inst
    2. evaluate
    """
    def __init__(self, sub_folder, args):
        super(EN_NER, self).__init__(sub_folder)

        self.task_name = 'EN_NER'
        self.prepare()
        self.definition = self.meta_data["defi"]
        self.n_examples = 20
        self.args = args

        self.head = "Below is an instruction that describes the {defi} task. Write a response that appropriately completes the request.\n".format(defi=self.definition)
        self.answer_template = """{out}"""

        self.prompt_template = self.head + self.meta_data["test_prompt"]["instruction"]


        self.label_sep = self.meta_data["label_sep"]
        self.output_template = lambda x: self.label_sep.join(x)
        self.label_key, self.label_list = self.meta_data['label_key'], self.meta_data["label_set"][self.meta_data['label_key']]

        print(self.task_name)


    def encode_to_input_output(self, example, args):
        """
        example -> instruction
        """
        inputs = example['sentences']
        type2spans = self.get_span_type_mapping(example)
        for k in range(self.n_per_sample):
            prompt = {'sentences': ''.join(inputs) if isinstance(inputs, list) else inputs,
                    'label_set': self.label_sep.join(self.meta_data["label_set"][self.meta_data["label_key"]]),
                      'ex': ''}
            output_ = []
            for i in self.label_list:
                output_ += [i + ': ' + self.label_sep.join(type2spans.get(i, ['None']))]
            output = "\n".join(output_)
            answer = {'out': output}
            all_prompt = [self.prompt_template.format(**prompt)]
            answer = self.answer_template.format(**answer)
            yield all_prompt, answer

            if len(pos_labels) == 1:
                break

    def save_data(self):
        for k, v in self.inst_data.items():
            save_path = os.path.join(self.data_folder,
                                     '{split}.{task}.inst.json'.format(split=k, task=self.task_name))
            with open(save_path, 'w') as f:
                dumped = json.dumps(v, indent=4 ,ensure_ascii=False)
                f.write(dumped)
            print("save data to: {}".format(save_path))
    def get_span_type_mapping(self, example):
        type2spans = {}
        for span in example["spans"]:
            if span["type"] not in type2spans:
                type2spans[span["type"]] = [span["term"]]
            else:
                type2spans[span["type"]].append(span["term"])
        return type2spans


    def parse_answer(self, answer):
        if len(answer) == 0:
            print("error")
        items = answer.split('\n')
        type_2_spans = dict()
        
        spilt_str = ": "
        for i in items:
            if spilt_str not in i or i.endswith(spilt_str):
                continue
            _type, spans = i[:i.index(spilt_str)], i[i.index(spilt_str)+len(spilt_str):].split(self.label_sep)
            if isinstance(spans, str): spans = [spans]
            if _type not in type_2_spans:
                type_2_spans[_type] = spans
            else:
                type_2_spans[_type] += spans

        for t, s in type_2_spans.items():
            type_2_spans[t] = set(s)
        if type_2_spans == {}:
            type_2_spans["None"] = ["None"]
        return type_2_spans

    def parse_glod(self, gold_answer):
        return self.parse_answer(gold_answer)

    def mapping2label(self, text, type2span):
        lt = len(text)
        labels = ['O']*lt
        for type, spans in type2span.items():
            for s in spans:
                if s == 'None':
                    continue
                else:
                    for j in range(len(text)):
                        if text[j: j + len(s)] == s:
                            labels[j] = 'B-' + type
                            labels[j+1: j+len(s)] = ['I-' + type]*(len(s)-1)
        return labels

    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']
            gold_answers.append(gold_answer)
            if len(pred_result)==0:
                pred_result = "error"
            pred_answers.append(pred_result)
            all_pred.append(self.mapping2label(d["sentences"], self.parse_answer(pred_result)))
            all_gold.append(self.mapping2label(d["sentences"], self.parse_glod(gold_answer)))
        scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
        print(all_gold)
        res = self.ext_scorer.evaluate(all_gold, all_pred)
        print(res)
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  res['micro avg']['f1-score'],
                      'macro-f1':  res['macro avg']['f1-score'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': res, 'rouge': scores}

        return detailed_report, key_scores

    def evaluate_old(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']
            gold_answers.append(gold_answer)
            if len(pred_result)==0:
                pred_result = "error"
            pred_answers.append(pred_result)
            all_pred.append(self.parse_answer(pred_result))
            all_gold.append(self.parse_glod(gold_answer))
        scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
        self.scorer.update(all_gold, all_pred)
        res = self.scorer.result()
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  res['micro_f1'],
                      'macro-f1':  res['macro_f1'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': res, 'rouge': scores}

        return detailed_report, key_scores