
import os 
from itertools import product

def run(bash_script_path: str,
        bash_script_args: str,
        job_name: str,
        log_file: str,
        n_tasks: str = '4',
        memory_gb: str = '5',
        time_hrs: str = '3',
        time_min: str = '00'):

    os.system('rm -f ' + log_file)

    #if log file has a folder, check if folder exists (if not, create folder)
    log_folder = os.path.dirname(log_file)
    if log_folder != '':
        if not os.path.isdir(f'logs/{log_folder}'):
            os.makedirs(f'logs/{log_folder}')

    sbatch_command = f'sbatch -p seas_gpu,gpu --gres=gpu:nvidia_a100-sxm4-80gb:1 --mem={memory_gb}gb --time=0-0{time_hrs}:00 --ntasks={n_tasks} --error=logs/{log_file} --output=logs/{log_file} --job-name={job_name}'
    # bash_script_path = 'inference_pipeline.sh'
    # bash_script_args = '-b pubmedqa -c meditron-7b -s 3 -m 1 -f False'

    full_cmd = f'{sbatch_command} {bash_script_path} {bash_script_args}'
    os.system(full_cmd)


### working example
# '-b pubmedqa -c meditron-7b -s 3 -m 1 -f False'
# run(bash_script_path='inference_pipeline.sh',
#     bash_script_args='-b pubmedqa -c medalpaca-7b -s 3 -m 1 -f False',
#     job_name='test',
#     log_file='logs/med-eval',
#     n_tasks='1',
#     memory_gb='30',
#     time_hrs='1',
#     time_min='00')


def dict2options(settings):
    keys = []
    opts = []
    for k in settings:
        keys.append(k)
        opts.append(settings[k])

    exp_list = list(product(*opts))

    options_str = []
    name_str = []
    for option in exp_list:
        temp_str = ''
        temp_name_str = ''
        for (k,i) in zip(keys, option):
            temp_str+=f'-{k} {i} '
            if k != 'save-path':
                temp_name_str += f'{i}_'
        options_str.append(temp_str)
        name_str.append(temp_name_str[0:-1])
    
    return options_str, name_str



###evaluate base models
def eval_base_models():

    settings = {
    'b': ['pubmedqa'], #benchmark dataset
    'c': ['medalpaca-7b'], #checkpoint, aka base model
    #arguments below stay constant
    's': ['3'], #number shots for in-context learning
    'm': ['1'], #whether multi-seed for in-context learning is on
    'f': ['False']
    }

    options_str, name_str = dict2options(settings)

    for (opt, name) in zip(options_str, name_str): 

        run(bash_script_path='inference_pipeline.sh', bash_script_args=opt, job_name=name,
            log_file=f'med_eval/log_{name}',
            n_tasks='1', memory_gb='30', time_hrs='1', time_min='00')

        print(f'job_name = {name}, options = {opt}')  



###evaluate fine-tuned models
def eval_ft_models():

    settings = {
    #arguments below vary
    'b': ['pubmedqa'], #benchmark dataset
    'c': ['medalpaca-7b'], #checkpoint, aka base model
    #arguments below stay constant
    's': ['3'], #number shots for in-context learning
    'm': ['1'], #whether multi-seed for in-context learning is on
    'f': ['True'], #whether model is fine-tuned
    #arguments below vary
    'h': ['gen'],
    'n': ['100']
    }

    options_str, name_str = dict2options(settings)

    for (opt, name) in zip(options_str, name_str): 

        run(bash_script_path='inference_pipeline.sh', bash_script_args=opt, job_name=name,
            log_file=f'med_eval/log_{name}',
            n_tasks='1', memory_gb='30', time_hrs='1', time_min='00')

        print(f'job_name = {name}, options = {opt}')  


### reproduce author results
def reproduce_author_results():

    #NOTE:
    #in inference.py and evaluate.py --> make sure you are using the authors original 3 seeds

    settings = {
    'b': ['pubmedqa', 'medmcqa', 'mmlu_medical', 'medqa4'], #benchmark dataset
    'c': ['meditron-7b'], #checkpoint, aka base model
    #arguments below stay constant
    's': ['3'], #number shots for in-context learning
    'm': ['1'], #whether multi-seed for in-context learning is on
    'f': ['False']
    }

    options_str, name_str = dict2options(settings)

    for (opt, name) in zip(options_str, name_str): 

        run(bash_script_path='inference_pipeline.sh', bash_script_args=opt, job_name=name,
            log_file=f'reproduce_author_results/log_{name}',
            n_tasks='1', memory_gb='30', time_hrs='1', time_min='00')

        print(f'job_name = {name}, options = {opt}')  



if __name__ == "__main__":
    # eval_base_models()
    # eval_ft_models()
    reproduce_author_results()