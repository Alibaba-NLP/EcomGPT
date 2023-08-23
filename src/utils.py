import sys
import json, os, importlib
import ast
sys.path.append('./data')
sys.path.append('../')
sys.path.append('../../')
import task_cls, task_extract

def get_cls(task_name, sub_folder, args):
    # modules = importlib.import_module("task_extract")
    # myclass = getattr(modules, task_name)(sub_folder,args)  
    # return myclass
    dir_path = "./data"
    files = [f for f in os.listdir(dir_path) if f.endswith('.py') and "task_" in f]    
    for f in files:    
        file_path = os.path.join(dir_path, f)
        file_path = file_path.replace(".py","").replace("./","").replace("/",".")
        print(file_path,"task_name")
        modules = importlib.import_module(file_path)
        
        try:
            myclass = getattr(modules, task_name)(sub_folder,args)  
            return myclass
        except:
            continue

    print(f"Failed to create instance of {task_name}")