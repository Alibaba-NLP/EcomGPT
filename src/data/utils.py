import sys
import json
import ast
sys.path.append('./data')
sys.path.append('../')
sys.path.append('../../')
# from src.meta import GEN_TOK, BOS_TOK, EOS_TOK, BASE_DATA_DIR

def load_data(file_path):
    d = []
    with open(file_path) as f:
        for line in f:
            try:
                js = json.loads(line)
            except:
                js = ast.literal_eval(line)
            d.append(js)
    return d


def wrap_answer(answer, tokenizer, add_gen=True):
    wrap = lambda answer: GEN_TOK + answer + tokenizer.eos_token if add_gen else answer + tokenizer.eos_token
    if isinstance(answer, list):
        return [wrap(i) for i in answer]
    else:
        return wrap(answer)

def wrap_answer_glm(answer, tokenizer, add_gen=True):
    wrap = lambda answer: tokenizer.gmask_token + answer + tokenizer.eos_token if add_gen else answer + tokenizer.eos_token
    if isinstance(answer, list):
        return [wrap(i) for i in answer]
    else:
        return wrap(answer)


def verify_data_name(name):
    extension = name.split(".")[-1]
    assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


def parse_data(data_str):
    data = data_str.split('[DATA]')
    data = [i for i in data if len(i) > 0]
    all_data = []
    for d in data:
        d_split = d.split('[SEP]')
        d_split = [i for i in d_split if len(i) > 1]
        if len(d_split) == 2:
            dname, portion = d.split('[SEP]')
            verify_data_name(dname)
            all_data.append((dname, float(portion)))
        elif len(d_split) == 1:
            dname = d_split[0]
            verify_data_name(dname)
            all_data.append((dname, 1.0))
        else:
            raise NotImplementedError("wrong data_file format")
    return all_data


def find_span_positions(text, span):
    ls = len(span)
    pos = []
    for j in range(len(text)):
        if text[j: j+ls] == span:
            pos.append((j, j+ls))
    return pos


def mapping2label(text, type2span):
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


def save_jsonline(data, f_path):
        with open(f_path, 'w') as f:
            dumped = [json.dumps(l, ensure_ascii=False) for l in data]
            for i in dumped[:-1]:
                f.write(i+'\n')
            f.write(dumped[-1])
        print("save data to: {}".format(f_path))
