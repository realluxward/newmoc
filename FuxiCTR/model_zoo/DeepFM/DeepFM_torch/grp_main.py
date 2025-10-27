

import os
import sys
import yaml
import itertools
import subprocess
from multiprocessing import Pool

def load_config(config_dir, expid):
    dataset_config_path = os.path.join(config_dir, 'dataset_config.yaml')
    model_config_path = os.path.join(config_dir, 'model_config.yaml')
    
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    exp_dataset_config = dataset_config.get(expid, {})
    base_model_config = model_config['base']
    exp_model_config = model_config.get(expid, {})
    
    config = {**exp_dataset_config, **base_model_config, **exp_model_config}
    return config

def save_config(config, output_dir, expid):
    config_path = os.path.join(output_dir, f"{expid}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path

def run_experiment(config_path, gpu):
    command = f"python run_expid.py --config {config_path} --gpu {gpu}"
    print(f"Running command: {command}") 
    subprocess.run(command, shell=True)
    
def run_single_experiment_worker(args):
    """
    Worker function to run a single experiment.
    Accepts a tuple (config_path, gpu_id) as its single argument.
    """
    config_path, gpu_id = args  # 解包参数
    run_experiment(config_path, gpu_id)

def main():
    base_config_dir = './config'
    output_dir = './yaml_moc_1025_experiments'
    os.makedirs(output_dir, exist_ok=True)

    expid = 'base'
    base_config = load_config(base_config_dir, expid)

    for seed in [20,201,1027,2024,2333][:3]:
        for lr in [0.001,0.005,0.01]:
            for dataset in ['beauty','sports','toys'][1:2]:
                for method in ['base','me', 'moc', 'rq'][1:]:
                    for scala in [0,1,3,7][1:]:
                        base_dataset_path = f'/data2/wangzhongren/taolin_project/data/{dataset}-split/base_dataset'
                        train_path = os.path.join(base_dataset_path, 'train.csv')
                        test_path = os.path.join(base_dataset_path, 'test.csv')
                        valid_path = os.path.join(base_dataset_path, 'valid.csv')
                        index_file_path = f'/data2/wangzhongren/taolin_project/dataset/{dataset}-split/{method}_cbsize256_cbdim32_scala{scala}_epoch500_index.pt'
                        config = base_config.copy()
                        config.update({ #'cov_weight': cov_weight,
                                        'train_data': train_path,
                                        'test_data': test_path,
                                        'valid_data': valid_path,
                                        'seed': seed,
                                        'model_root': 'moc_1025_experiments',
                                        'dataset_id': dataset,
                                        'batch_size': 10000, #4096
                                        'learning_rate':lr,
                                        'use_index_emb': True,
                                        'index_file_path': index_file_path,
                                    })
                        config['expid'] = f"{dataset}_{method}_scala{scala}_seed{seed}_lr{lr}" 
                        save_config(config, output_dir, config['expid'])
    config_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.yaml')]

    NUM_GPUS = 8
    NUM_PROCESSES = 16

    job_args = []
    for i, config_file in enumerate(config_files):
        gpu_id = i % NUM_GPUS  # 使用取模运算实现轮询
        job_args.append((config_file, gpu_id))

    if not job_args:
        print("未找到任何实验配置，程序退出。")
        return

    print("\n" + "="*60)
    print("            EXPERIMENT BATCH SUMMARY")
    print("="*60)
    print(f"将要启动的实验总数: {len(job_args)}")
    print(f"将要使用的GPU数量: {NUM_GPUS}")
    print(f"将要启动的并行进程数: {NUM_PROCESSES}")
    print(f"配置文件输出目录: {output_dir}")
    print("-"*60)
    print("前3个任务示例:")
    for cfg_path, gpu in job_args[:3]:
        print(f"  - GPU {gpu}: python run_expid.py --config {cfg_path}")
    print("="*60)

    confirmation = input("\n>>> 是否要启动以上所有实验? (输入 'y' 或 'yes' 确认): ")

    # 4. 根据用户输入决定是否执行
    if confirmation.lower().strip() in ['y', 'yes']:
        print("\n--- 用户已确认，开始并行执行所有实验... ---")
        with Pool(processes=NUM_PROCESSES) as pool:
            pool.map(run_single_experiment_worker, job_args)
        print("\n--- 所有实验任务已分发完毕 ---")
    else:
        print("\n--- 用户取消操作，未执行任何实验 ---")

if __name__ == '__main__':
    main()

