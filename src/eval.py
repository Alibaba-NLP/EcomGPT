from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import argparse
from datetime import datetime
import os,json,sys
sys.path.append('.')
from meta import GEN_TOKEN
from utils import get_cls
import random
from tqdm import tqdm
import torch
from data.utils import save_jsonline, load_data
from glob import glob
import re

def generate_prompts(p, model, tokenizer, gpu=1, prompt_length=256, answer_length=44, beam_size=4):
    tokenizer.truncation_side = 'left'
    tokenizer.padding_side = 'left' 
    # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    input_ids = tokenizer(p,
                          return_tensors="pt",
                          padding=True,
                          truncation=True,
                          max_length=prompt_length).input_ids
    if torch.cuda.is_available() and gpu:
        input_ids = input_ids.cuda()
        model.cuda()
    with torch.no_grad():
        outputs = model.generate(input_ids, num_beams=beam_size, do_sample=False, max_length=answer_length+prompt_length)
    outputs_answer = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    all_answer = []
    for i in outputs_answer:
        if (GEN_TOKEN in i) and (tokenizer.eos_token in i):
            gen_idx = i.index(GEN_TOKEN) + len(GEN_TOKEN)
            end_idx = i.index(tokenizer.eos_token)
            all_answer.append(i[gen_idx: end_idx])
        elif (GEN_TOKEN in i) and (tokenizer.eos_token not in i):
            gen_idx = i.index(GEN_TOKEN) + len(GEN_TOKEN)
            all_answer.append(i[gen_idx:])
        else:
            all_answer.append('')
    return [i.replace(tokenizer.pad_token, '') for i in all_answer]


def eval(args):
    base_save_dir = args.base_save_dir
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M-%s")
    if "checkpoint" in args.model_name_or_path:
        model_dir, ckp = os.path.split(os.path.normpath(args.model_name_or_path))
        m_str = os.path.join(os.path.basename(model_dir), ckp)
    else:
        m_str = os.path.basename(os.path.normpath(args.model_name_or_path))
    dataset_name = args.dataset_name.replace('/', '-')
    save_dir = os.path.join(base_save_dir, args.save_name, m_str)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '{}_{}_{}.beam{}.res'.format(dataset_name, args.sub_folder, args.task, args.beam_size))
    print('saving at: {}'.format(save_path))
    print("use GPU: {}".format(torch.cuda.is_available()))

    print('dataset_name', args.dataset_name)
    print('subfolder', args.sub_folder)

    eval_data_path = os.path.join(args.base_data_dir, args.dataset_name, "tasks", args.sub_folder, 'test.json')
    LANG, _,TASK_TYPE,_=args.sub_folder.split("-")
    eval_data_save_path = os.path.join(args.base_data_dir, args.dataset_name, "tasks", args.sub_folder, "test.{}_{}.inst.json".format(LANG,TASK_TYPE))


    task = get_cls(args.task, os.path.join(args.base_data_dir, args.dataset_name, "tasks", args.sub_folder), args)


    if os.path.exists(eval_data_path):
        print('loading eval data: {}'.format(eval_data_path))
        sample_data = load_data(eval_data_path)[:5]
        for idx, i in enumerate(sample_data):
            inp, out = next(task.encode_to_input_output(i, args))
            i['prompt'] = inp
            i['answer'] = out
            i['idx'] = str(idx)+"-"+args.sub_folder.replace("/","")

    else:
        raise NotImplementedError('no eval data')

            
    print(sample_data[0]['prompt'])
    print(sample_data[0]['answer'])


    p_single = [i['prompt'][0] for i in sample_data if len(i['prompt']) == 1]
    sample_single = [i for i in sample_data if len(i['prompt']) == 1]


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.eval()

    def iter_prompt_list(p_list):
        res = []
        l = 0
        r = l + args.batch_size
        while l < len(p_list):
            prompt_list = p_list[l: r]
            tmp_res = generate_prompts(prompt_list, model, tokenizer, args.gpu,
                                    args.prompt_length, args.answer_length, args.beam_size)
            res += tmp_res
            idx = 0
            l = r
            r = r + args.batch_size
            for i in tmp_res:
                print('EVAL: {} ================================================'.format(idx))
                print(prompt_list[idx])
                print(i)
                f.write(prompt_list[idx])
                f.write(i)
                f.write('\n======================================================\n')
                idx += 1

        return res

    key_result_file = os.path.join(save_dir,"{}_results.txt".format(os.path.basename(os.path.normpath(args.model_name_or_path))))
    with open(save_path, 'w') as f, open(key_result_file, 'a') as fin:
        print("{} SINGLE LIST, BATCH EVAL".format(len(p_single)))
        res_new = iter_prompt_list(p_single)
        print("TOTAL RES NUMBER: {}".format(len(res_new)))
        res_detail, res_key = task.evaluate(sample_single, res_new)
        print(res_detail)
        for k, v in res_detail.items():
            f.write(k + ' ============ \n')
            f.write(str(v) + '\n')
        res_detail['model'] = args.model_name_or_path
        res_detail['data'] = args.dataset_name
        res_detail['beam'] = args.beam_size
        print(res_key)
        f.write(str(res_detail))
        for k,v in res_key.items():
            fin.write("-".join([dataset_name, args.sub_folder, args.task])+"\t"+str(v)+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='ZH_NER_OPEN')
    parser.add_argument('-tf', '--task_file', type=str, default='NA')
    parser.add_argument('-sf', '--sub_folder', type=str, default='NER')
    parser.add_argument('-d', '--dataset_name', type=str, default='all')
    parser.add_argument('-m', '--model_name_or_path', type=str, default='../output/2-24-bigscience/bloom-560m/')
    parser.add_argument('-bsd', '--base_save_dir', type=str, default='../output/evaluate')
    parser.add_argument('-bdd', '--base_data_dir', type=str, default='')
    parser.add_argument('-bz', '--batch_size', type=int, default=2)
    parser.add_argument('-pl', '--prompt_length', type=int, default=800)
    parser.add_argument('-al', '--answer_length', type=int, default=164)
    parser.add_argument('-beam', '--beam_size', type=int, default=4)
    parser.add_argument('-sn', '--save_name', type=str, default='6-28')
    parser.add_argument('-g', '--gpu', type=int, default=1)

    random.seed(42)
    args = parser.parse_args()
    with open(args.task_file) as f:
        tasks = f.read().strip().split("\n")
    task_type2task_class = {"CLS": "EN_CLS", "Extract": "EN_EXTRACT", "Generate": "EN_GEN", \
        "Condition_Extract": "EN_CONDITION_EXTRACT", "NER": "EN_NER"}
    for task in tasks:
        args.sub_folder = task
        args.dataset_name = task.split("-")[1]
        args.task = task_type2task_class[task.split("-")[2]]
        eval(args)