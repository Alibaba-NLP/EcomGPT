import json
import random
import os
from tqdm import tqdm
import sys
sys.path.append('../')
sys.path.append('../../')
from data.utils import load_data, mapping2label
from seqeval.metrics import classification_report
from collections import Counter
import numpy as np
from rouge import Rouge

from src.meta import BASE_DATA_DIR


class Task:
    def __init__(self, sub_folder, dataset_name):
        self.task_name = 'NER'
        self.definition = 'Named Entity Recognition'
        self.n_examples = 20
        self.sub_folder = '{}/{}/'.format(sub_folder, dataset_name)
        self.train_file = 'train.json'
        self.val_file = 'test.json'
        self.data_folder = os.path.join(BASE_DATA_DIR, self.sub_folder)
        self.n_per_sample = 1

        self.prompt_template = """Given the sentence {inp} \n
                            Show me the named entities and their types \n"""
        self.answer_template = """{out}"""
        self.overall_template = "{def} \n {ex} \n {conj} \n {prompt}"
        self.example_template = "Example {i}: \n Input: {inp}\nOutput: {out}"

        self.train_data = None
        self.val_data = None
        self.example_data = None
        self.inst_data = None


    def get_example(self):
        for i in range(3):
            print('EXAMPLE {}'.format(i))
            inp, out = next(self.encode_to_input_output(self.train_data[i]))
            prompt = self.prompt_template.format(**inp)
            answer = self.answer_template.format(**out)
            print("prompt:\n{}\nanswer:\n{}".format(prompt, answer))

    def prepare(self):
        self.train_data = load_data(os.path.join(os.path.join(BASE_DATA_DIR, self.sub_folder), self.train_file))
        self.val_data = load_data(os.path.join(os.path.join(BASE_DATA_DIR, self.sub_folder), self.val_file))
        random.shuffle(self.train_data)
        self.example_data = self.train_data[: self.n_examples]
        self.train_data = self.train_data[self.n_examples:]


    def encode_to_input_output(self, example):
        yield dict(), dict()

    def generate(self):
        train_data = []
        for e in tqdm(self.train_data, desc='generate training data'):
            for inp, out in self.encode_to_input_output(e):
                prompt = self.prompt_template.format(**inp)
                answer = self.answer_template.format(**out)
                train_data.append({'prompt': prompt, 'answer': answer})

        val_data = []
        for e in tqdm(self.val_data, desc='generate validation data'):
            for inp, out in self.encode_to_input_output(e):
                prompt = self.prompt_template.format(**inp)
                answer = self.answer_template.format(**out)
                val_data.append({'prompt': prompt, 'answer': answer})

        self.inst_data = {'train': train_data, 'val': val_data}

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

    def save_data(self):
        for k, v in self.inst_data.items():
            save_path = os.path.join(os.path.join(os.path.join(BASE_DATA_DIR, self.sub_folder)),
                                     '{split}.{task}.inst.json'.format(split=k, task=self.task_name))
            with open(save_path, 'w') as f:
                dumped = [json.dumps(l, ensure_ascii=False) for l in v]
                for i in dumped[:-1]:
                    f.write(i+'\n')
                f.write(dumped[-1])
            print("save data to: {}".format(save_path))

    def evaluate(self, data, results):
        pass

    def parse_answer(self, answer):
        pass


class ZH_NER_OPEN(Task):
    def __init__(self, sub_folder, dataset_name):
        super().__init__(sub_folder, dataset_name)
        self.task_name = 'ZH_NER_OPEN'
        self.definition = '中文命名体识别'
        self.train_file = 'train.json'
        self.val_file = 'test.json'
        self.n_examples = 20
        self.train_data = None
        self.val_data = None
        self.example_data = None
        self.inst_data = None

        self.prompt_template = """给以下句子 {inp} \n 请标注出其中的实体以及他们的类型 \n"""
        self.answer_template = """{out}"""

        self.prepare()
        print(self.task_name)


    def encode_to_input_output(self, example):
        inputs = example['text']
        spans = example['label'] if 'label' in example else example['spans']

        answer = ''
        for s in spans:
            span_str = inputs[s['start']: s['end']]
            if isinstance(span_str, list): span_str = ''.join(span_str)  # list of tokens, ZH
            answer += '{span}: {type}\n'.format(span=span_str, type='\t'.join(s['type']) if isinstance(s['type'], list) else s['type'])
        if answer == '':
            answer = 'None'
        yield {'inp': ''.join(inputs) if isinstance(inputs, list) else inputs}, {'out': answer}


class ZH_NER_TYPE_SPAN(Task):
    def __init__(self, sub_folder, dataset_name):
        super().__init__(sub_folder, dataset_name)

        self.task_name = 'ZH_NER_TYPE_SPAN'
        self.definition = '中文命名实体查找'
        self.train_file = 'train.json'
        self.val_file = 'test.json'
        self.n_examples = 20
        self.train_data = None
        self.val_data = None
        self.example_data = None
        self.inst_data = None

        self.prompt_template = """给以下句子 {sent}, 请找到其中所有的 {label} \n"""
        self.answer_template = """{out}"""

        self.prepare()

        self.all_types = self.get_all_label()
        self.in_sample_rate = 0.8
        print(self.task_name)

    def get_all_label(self):
        all_types = Counter()
        for e in tqdm(self.train_data, desc='get all labels in the training'):
            spans = e['label'] if 'label' in e else e['spans']
            # get span2types and type2spans
            for s in spans:
                span_type = s['type']
                # span type should be str or list
                span_type = [span_type] if isinstance(span_type, str) else span_type
                span_type = list(set(span_type))
                if len(span_type) == 0:
                    continue
                all_types.update(span_type)
        return all_types

    def get_span_type_mapping(self, inputs, spans):
        span2types = dict()
        type2spans = dict()

        # get span2types and type2spans
        for s in spans:
            span_str = inputs[s['start']: s['end']]
            if isinstance(span_str, list): span_str = ''.join(span_str)  # list of tokens, ZH
            span_type = s['type']
            # span type should be str or list
            span_type = [span_type] if isinstance(span_type, str) else span_type
            if span_str not in span2types: span2types[span_str] = span_type
            else: span2types[span_str] += span_type
            random.shuffle(span_type)
            for t in span_type[:1]:
                if t not in type2spans: type2spans[t] = [span_str]
                else: type2spans[t].append(span_str)
        return span2types, type2spans


    def encode_to_input_output(self, example):
        inputs = example['text']
        spans = example['label'] if 'label' in example else example['spans']

        span2types, type2spans = self.get_span_type_mapping(inputs, spans)

        # in sample label
        in_sample_label_list = list(type2spans.keys())
        # out sample label
        all_types = example.get('label_set', self.all_types)
        out_sample_label_list = list(set(all_types) - set(list(type2spans.keys())))

        if out_sample_label_list:
            label = random.choice(out_sample_label_list)
            spans = ['None']
            prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs, 'label': label}
            answer = {'out': '\t'.join(spans)}
            yield prompt, answer

        if in_sample_label_list:
            for label in in_sample_label_list:
                spans = type2spans[label]
                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs, 'label': label}
                answer = {'out': '\t'.join(spans)}
                yield prompt, answer



class ZH_NER(ZH_NER_TYPE_SPAN):
    def __init__(self, sub_folder, dataset_name):
        super(ZH_NER, self).__init__(sub_folder, dataset_name)

        self.task_name = 'ZH_NER'
        self.definition = '中文命名实体识别'
        self.train_file = 'train.json'
        self.val_file = 'test.json'
        self.n_examples = 20
        self.train_data = None
        self.val_data = None
        self.example_data = None
        self.inst_data = None

        self.prompt_template = """给以下句子 {sent}, 请找到其中所有的 {label} \n"""
        self.answer_template = """{out}"""

        self.prepare()
        self.all_types = self.get_all_label()
        self.max_top_label = 5
        self.max_remain_label = 20
        self.max_pos_label = 20
        self.n_per_sample = 3
        self.top_sqrt = 0.5
        self.top_K = self.all_types.most_common(500)
        print(self.task_name)

        self.dict_top_k = dict(self.top_K)
        self.list_top_k = [k for k, v in self.top_K]
        self.set_top_k = {k for k, v in self.top_K}
        self.remain = list(set(self.all_types.keys()) - self.set_top_k)


    def encode_to_input_output(self, example):
        inputs = example['text']
        spans = example['label'] if 'label' in example else example['spans']

        for k in range(3):
            span2types, type2spans = self.get_span_type_mapping(inputs, spans)
            # in sample label
            in_sample_label_list = list(type2spans.keys())
            random.shuffle(in_sample_label_list)
            # out sample label
            all_types = example.get('label_set', self.all_types)
            if 'label_set' not in example:
                all_types = example.get('label_list', self.all_types)

            for j in range(self.n_per_sample):

                if ('label_set' in example) or ('label_list' in example):
                    sampled_label = all_types
                else:
                    out_sample_label_list_k = list(self.set_top_k - set(in_sample_label_list))
                    freq_list = np.array([self.dict_top_k[k] for k in out_sample_label_list_k])**self.top_sqrt
                    prob_list = freq_list / freq_list.sum()
                    n_out_top = np.random.randint(self.max_top_label)
                    n_out_remain = np.random.randint(self.max_remain_label)
                    n_pos = np.random.randint(self.max_pos_label)
                    types_n_top_k = np.random.choice(out_sample_label_list_k, p=prob_list, size=n_out_top, replace=True)
                    types_n_remain = random.choices(self.remain, k=n_out_remain)
                    types_pos = in_sample_label_list[:n_pos]
                    sampled_label = list(set(types_n_top_k.tolist() + types_n_remain + types_pos))

                random.shuffle(sampled_label)
                label = ','.join(sampled_label)
                out = ''
                for i in sampled_label:
                    out += (i + ':' + '\t'.join(type2spans.get(i, ['None'])) + '\n')

                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs, 'label': label}
                answer = {'out': out}
                yield prompt, answer

    def parse_answer(self, answer):
        if len(answer) == 0:
            print('error')
        items = answer.split('\n')
        type_2_spans = dict()

        for i in items:
            if ':' not in i or i[-1] == ':':
                continue
            _type, spans = i[:i.index(':')], i[i.index(':')+1:].split('\t')
            if isinstance(spans, str): spans = [spans]
            if _type not in type_2_spans:
                type_2_spans[_type] = spans
            else:
                type_2_spans[_type] += spans

        for t, s in type_2_spans.items():
            type_2_spans[t] = set(s)
        return type_2_spans


    # def evaluate(self, data, results):
    #     all_gold = []
    #     all_pred = []
    #     gold_answers = []
    #     pred_answers = []
    #     rouge = Rouge()
    #     for d, pred_result in zip(data, results):
    #         prompt = d['prompt']
    #         gold_answer = d['answer']
    #         example = d['example']
    #         inputs = example['text']
    #         text = ''.join(inputs) if isinstance(inputs, list) else inputs
    #         gold_type_2_spans = self.parse_answer(gold_answer)
    #         pred_type_2_spans = self.parse_answer(pred_result[len(prompt):])
    #         if len(gold_answer) and len(pred_result[len(prompt):]):
    #             gold_answers.append(gold_answer)
    #             pred_answers.append(pred_result[len(prompt):])
    #
    #         gold_labels = mapping2label(text, gold_type_2_spans)
    #         pred_labels = mapping2label(text, pred_type_2_spans)
    #         all_gold.append(gold_labels)
    #         all_pred.append(pred_labels)
    #     report = classification_report(all_gold, all_pred, output_dict=True)
    #
    #     scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
    #     macro_score = report['macro avg']
    #     micro_score = report['micro avg']
    #     rouge_l_score = scores['rouge-l']
    #     key_scores = {'micro-f1':  micro_score['f1-score'],
    #                   'macro-f1':  macro_score['f1-score'],
    #                   'rouge-l-f1': rouge_l_score['f']}
    #     detailed_report = {'seqeval': report, 'rouge': scores}
    #
    #     return detailed_report, key_scores


    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']
            example = d['example']
            inputs = example['text']
            text = ''.join(inputs) if isinstance(inputs, list) else inputs
            gold_type_2_spans = self.parse_answer(gold_answer)
            pred_type_2_spans = self.parse_answer(pred_result)
            if len(gold_answer) and len(pred_result):
                gold_answers.append(gold_answer)
                pred_answers.append(pred_result)
            else:
                print('miss')

            gold_labels = mapping2label(text, gold_type_2_spans)
            pred_labels = mapping2label(text, pred_type_2_spans)
            all_gold.append(gold_labels)
            all_pred.append(pred_labels)
        report = classification_report(all_gold, all_pred, output_dict=True)

        scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
        macro_score = report['macro avg']
        micro_score = report['micro avg']
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  micro_score['f1-score'],
                      'macro-f1':  macro_score['f1-score'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': report, 'rouge': scores}

        return detailed_report, key_scores



class ZH_NER_ICL(ZH_NER_TYPE_SPAN):
    def __init__(self, sub_folder, dataset_name):
        super(ZH_NER_ICL, self).__init__(sub_folder, dataset_name)

        self.task_name = 'ZH_NER_ICL'
        self.definition = '中文命名实体识别'
        self.train_file = 'train.json'
        self.val_file = 'test.json'
        self.n_examples = 20
        self.train_data = None
        self.val_data = None
        self.example_data = None
        self.inst_data = None

        self.answer_template = """{out}"""
        self.prompt_template = "我们现在做中文命名实体的抽取，\n{ex}接下来给定以下\n输入： {sent}，抽取所有的{label}\n输出："
        self.example_template = "例子 {i}：\n输入: {sent}，抽取所有的{label}\n输出: {out}\n"
        self.output_template = lambda x: "，".join(x)

        self.prepare()
        self.all_types = self.get_all_label()
        self.all_type_sample_mapping = self.get_type_sample_mapping()
        self.max_top_label = 5
        self.max_remain_label = 20
        self.max_pos_label = 20
        self.n_per_sample = 3
        self.n_repeat = 3
        self.top_sqrt = 0.5
        self.add_example_prob = 0.2
        self.max_example = 3
        self.top_K = self.all_types.most_common(500)
        print(self.task_name)

        self.dict_top_k = dict(self.top_K)
        self.list_top_k = [k for k, v in self.top_K]
        self.set_top_k = {k for k, v in self.top_K}
        self.remain = list(set(self.all_types.keys()) - self.set_top_k)


    def get_type_sample_mapping(self):
        type_sample_mapping = dict()
        id = 0
        for e in tqdm(self.train_data, desc='get all type sample mapping in the training'):
            spans = e['label'] if 'label' in e else e['spans']
            # get span2types and type2spans
            for s in spans:
                span_type = s['type']
                # span type should be str or list
                span_type = [span_type] if isinstance(span_type, str) else span_type
                span_type = list(set(span_type))
                if len(span_type) == 0:
                    continue
                else:
                    for t in span_type:
                        if t not in type_sample_mapping:
                            type_sample_mapping[t] = [id]
                        else:
                            type_sample_mapping[t].append(id)
            id += 1
        return type_sample_mapping


    def encode_to_input_output(self, example):
        inputs = example['text']
        spans = example['label'] if 'label' in example else example['spans']

        for k in range(3):
            span2types, type2spans = self.get_span_type_mapping(inputs, spans)
            # in sample label
            in_sample_label_list = list(type2spans.keys())
            random.shuffle(in_sample_label_list)
            # out sample label
            all_types = example.get('label_set', self.all_types)
            if 'label_set' not in example:
                all_types = example.get('label_list', self.all_types)
            for j in range(self.n_per_sample):
                if ('label_set' in example) or ('label_list' in example):
                    sampled_label = all_types
                else:
                    sampled_label = self.sample_label(in_sample_label_list)

                random.shuffle(sampled_label)
                label = self.output_template(sampled_label)
                out = ''
                for i in sampled_label:
                    out += (i + ':' + '\t'.join(type2spans.get(i, ['None'])) + '\n')

                prompt = {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                          'label': label,
                          'ex': self.encode_to_example(self.sample_examples(type2spans))}
                answer = {'out': out}
                yield prompt, answer


    def encode_to_input_output_ex(self, example):
        inputs = example['text']
        spans = example['label'] if 'label' in example else example['spans']

        span2types, type2spans = self.get_span_type_mapping(inputs, spans)
        # in sample label
        in_sample_label_list = list(type2spans.keys())
        random.shuffle(in_sample_label_list)
        # out sample label
        all_types = example.get('label_set', self.all_types)
        if 'label_set' not in example:
            all_types = example.get('label_list', self.all_types)
        if ('label_set' in example) or ('label_list' in example):
            sampled_label = all_types
        else:
            sampled_label = self.sample_label(in_sample_label_list)

        random.shuffle(sampled_label)
        label = self.output_template(sampled_label)
        out = ''
        for i in sampled_label:
            out += (i + ':' + '\t'.join(type2spans.get(i, ['None'])) + '\n')

        return {'sent': ''.join(inputs) if isinstance(inputs, list) else inputs,
                'label': label, 'out': out}


    def sample_label(self, in_sample_label_list):
        out_sample_label_list_k = list(self.set_top_k - set(in_sample_label_list))
        freq_list = np.array([self.dict_top_k[k] for k in out_sample_label_list_k]) ** self.top_sqrt
        prob_list = freq_list / freq_list.sum()
        n_out_top = np.random.randint(self.max_top_label)
        n_out_remain = np.random.randint(self.max_remain_label)
        n_pos = np.random.randint(self.max_pos_label)
        types_n_top_k = np.random.choice(out_sample_label_list_k, p=prob_list, size=n_out_top, replace=True)
        types_n_remain = random.choices(self.remain, k=n_out_remain)
        types_pos = in_sample_label_list[:n_pos]
        sampled_label = list(set(types_n_top_k.tolist() + types_n_remain + types_pos))
        return sampled_label


    def sample_examples(self, type2spans):
        # sample examples
        pos_types = list(type2spans.keys())
        random.shuffle(pos_types)
        all_example = []
        for i in range(self.max_example):
            if random.random() < self.add_example_prob:
                pos_t = random.choice(pos_types)  # sample from a pos related examples
                if pos_t in self.all_type_sample_mapping:
                    example = self.train_data[random.choice(self.all_type_sample_mapping[pos_t])]
                else:
                    example = random.choice(self.train_data)
                all_example.append(example)
        return all_example

    def encode_to_example(self, examples):
        # sample examples
        all_ex = ''
        id = 0
        for i in examples:
            id += 1
            _ex = self.encode_to_input_output_ex(i)
            _ex['i'] = id
            ex = self.example_template.format(**_ex)
            all_ex += ex
        return all_ex

    def parse_answer(self, answer):
        if len(answer) == 0:
            print('error')
        items = answer.split('\n')
        type_2_spans = dict()

        for i in items:
            if ':' not in i or i[-1] == ':':
                continue
            _type, spans = i[:i.index(':')], i[i.index(':')+1:].split('\t')
            if isinstance(spans, str): spans = [spans]
            if _type not in type_2_spans:
                type_2_spans[_type] = spans
            else:
                type_2_spans[_type] += spans

        for t, s in type_2_spans.items():
            type_2_spans[t] = set(s)
        return type_2_spans


    def evaluate(self, data, results):
        all_gold = []
        all_pred = []
        gold_answers = []
        pred_answers = []
        rouge = Rouge()
        for d, pred_result in zip(data, results):
            gold_answer = d['answer']
            example = d['example']
            inputs = example['text']
            text = ''.join(inputs) if isinstance(inputs, list) else inputs
            gold_type_2_spans = self.parse_answer(gold_answer)
            pred_type_2_spans = self.parse_answer(pred_result)
            if len(gold_answer) and len(pred_result):
                gold_answers.append(gold_answer)
                pred_answers.append(pred_result)
            else:
                print('miss')

            gold_labels = mapping2label(text, gold_type_2_spans)
            pred_labels = mapping2label(text, pred_type_2_spans)
            all_gold.append(gold_labels)
            all_pred.append(pred_labels)
        report = classification_report(all_gold, all_pred, output_dict=True)

        scores = rouge.get_scores(pred_answers, gold_answers, avg=True)
        macro_score = report['macro avg']
        micro_score = report['micro avg']
        rouge_l_score = scores['rouge-l']
        key_scores = {'micro-f1':  micro_score['f1-score'],
                      'macro-f1':  macro_score['f1-score'],
                      'rouge-l-f1': rouge_l_score['f']}
        detailed_report = {'seqeval': report, 'rouge': scores}

        return detailed_report, key_scores


