# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:41:07 2023

@author: Sen
"""


import os
import sys
import subprocess
import hashlib
import warnings
import platform
import csv
import numpy as np
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import shutil

from openbabel import openbabel
import logging
import time
import subprocess
import threading
import os
import signal
import psutil

import os

class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            try:
                if os.name == 'posix':  # Unix/Linux/Mac
                    self.process = subprocess.Popen(self.cmd, shell=True, stderr=subprocess.DEVNULL,preexec_fn=os.setsid)
                else:  # Windows
                    self.process = subprocess.Popen(self.cmd, shell=True, stderr=subprocess.DEVNULL)
                self.process.communicate()
            except Exception:
                pass

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            if os.name == 'posix':  # Unix/Linux/Mac
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:  # Windows
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            thread.join()
        return self.process.returncode if self.process else None


class LigandPostprocessor:
    def __init__(self, path):
        self.hash_ligand_mapping = {}
        self.output_path = path  # Output directory for SDF files
        self.load_mapping()

    def load_mapping(self):
        mapping_file = os.path.join(output_path, 'hash_ligand_mapping.csv')
        if os.path.exists(mapping_file):
            print("Found existed mapping file, now reading ...")
            with open(mapping_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.hash_ligand_mapping[row[0]] = row[1]

    # Define a function to save the hash-ligand mapping to a file
    def save_mapping(self):
        mapping_file = os.path.join(output_path, 'hash_ligand_mapping.csv')
        with open(mapping_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for ligand_hash, ligand in self.hash_ligand_mapping.items():
                writer.writerow([ligand_hash, ligand])

    # Define a function to filter out empty SDF files
    def filter_sdf(self, hash_ligand_mapping_per_batch):
        print("Filtering sdf ...")
        ligand_hash_list = list(hash_ligand_mapping_per_batch.keys())
        mapping_per_match = hash_ligand_mapping_per_batch.copy()
        for ligand_hash in tqdm(ligand_hash_list):
            filepath = os.path.join(self.output_path, ligand_hash + '.sdf')            
            if os.path.getsize(filepath) < 2*1024:  #2kb
                try:
                    os.remove(filepath)
                    #mapping_per_match.pop(ligand_hash)
                except Exception:
                    print(filepath)
                mapping_per_match.pop(ligand_hash)    
        return mapping_per_match

    # Define a function to generate SDF files from a list of ligand SMILES using OpenBabel
    def to_sdf(self, ligand_list_per_batch):
        print("Converting to sdf ...")
        hash_ligand_mapping_per_batch = {}

        for ligand in tqdm(ligand_list_per_batch):
            ligand = ligand.strip()
            if not ligand:
                continue

            # 只取第一段，避免模型在 <L> 后面带额外文本
            ligand = ligand.split()[0]

            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("smi", "smi")
            mol = openbabel.OBMol()
            if not obConversion.ReadString(mol, ligand):
                print(f"[skip] invalid smiles: {ligand!r}")
                continue

            num_atoms = sum(1 for atom in openbabel.OBMolAtomIter(mol) if atom.GetAtomicNum() != 1)
            if min_atoms is not None and num_atoms < min_atoms:
                continue
            if max_atoms is not None and num_atoms > max_atoms:
                continue

            ligand_hash = hashlib.sha1(ligand.encode()).hexdigest()
            if ligand_hash in self.hash_ligand_mapping:
                continue

            filepath = os.path.join(self.output_path, ligand_hash + ".sdf")
            obabel_path = shutil.which("obabel")
            if not obabel_path:
                print("[error] obabel not found in PATH")
                continue

            cmd = [
                obabel_path,
                f"-:{ligand}",
                "-osdf",
                "-O", filepath,
                "--gen3d",
                "--forcefield", "mmff94",
            ]

            try:
                res = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                    shell=False,
                )
                if res.returncode != 0:
                    print(f"[obabel fail] rc={res.returncode} ligand={ligand!r}")
                    print(res.stderr[:300] if res.stderr else "[no stderr]")
                    continue
            except subprocess.TimeoutExpired:
                print(f"[obabel timeout] ligand={ligand!r}")
                continue
            except Exception as e:
                print(f"[obabel exception] ligand={ligand!r} err={e}")
                continue

            if os.path.exists(filepath):
                hash_ligand_mapping_per_batch[ligand_hash] = ligand

        self.hash_ligand_mapping.update(self.filter_sdf(hash_ligand_mapping_per_batch))    

    def delete_empty_files(self):
    # 遍历指定目录及其子目录中的所有文件
        for foldername, subfolders, filenames in os.walk(self.output_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                # 如果文件大小为0，则删除该文件
                if os.path.getsize(file_path) < 2*1024:  #2kb
                    try:
                        os.remove(file_path)
                        print(f'Deleted {file_path}')
                    except Exception:
                        pass 
    
    
    def check_sdf(self):
        file_list = os.listdir(self.output_path)
        sdf_file_list = [x for x in file_list if x[-4:]=='sdf']
        for filename in sdf_file_list:
            hash_ = filename[:-4]
            if hash_ not in self.hash_ligand_mapping.keys():
                filepath = os.path.join(self.output_path,filename)
                try:
                    os.remove(filepath)
                    print('remove ' + filepath)
                except Exception:
                    pass
            else:pass    
                
               
                
    
def about():
    print("""
  _____                    _____ _____ _______ 
 |  __ \                  / ____|  __ \__   __|
 | |  | |_ __ _   _  __ _| |  __| |__) | | |   
 | |  | | '__| | | |/ _` | | |_ |  ___/  | |   
 | |__| | |  | |_| | (_| | |__| | |      | |   
 |_____/|_|   \__,_|\__, |\_____|_|      |_|   
                     __/ |                     
                    |___/                      
 A generative drug design model based on GPT2
    """)


# Function to read in FASTA file
def read_fasta_file(file_path):
    with open(file_path, 'r') as f:
        sequence = []

        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                sequence.append(line)

        protein_sequence = ''.join(sequence)
    return protein_sequence


                    
if __name__ == "__main__":
    about()
    warnings.filterwarnings('ignore')
    
    if platform.system() == "Linux":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    #Sometimes, using Hugging Face may require a proxy.
    #os.environ["http_proxy"] = "http://your.proxy.server:port"
    #os.environ["https_proxy"] = "http://your.proxy.server:port"

    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pro_seq', type=str, default=None, help='Input a protein amino acid sequence. Default value is None. Only one of -p and -f should be specified.')
    parser.add_argument('-f','--fasta', type=str, default=None, help='Input a FASTA file. Default value is None. Only one of -p and -f should be specified.')
    parser.add_argument('-l','--ligand_prompt', type=str, default='', help='Input a ligand prompt. Default value is an empty string.')
    parser.add_argument('-e','--empty_input', action='store_true', default=False, help='Enable directly generate mode.')
    parser.add_argument('-n','--number',type=int, default=100, help='At least how many molecules will be generated. Default value is 100.')
    parser.add_argument('-d','--device',type=str, default='cuda', help="Hardware device to use. Default value is 'cuda'.")
    parser.add_argument('-o','--output', type=str, default='./ligand_output/', help="Output directory for generated molecules. Default value is './ligand_output/'.")
    parser.add_argument('-b','--batch_size', type=int, default=16, help="How many molecules will be generated per batch. Try to reduce this value if you have low RAM. Default value is 16.")
    parser.add_argument('-t','--temperature', type=float, default=1.0, help="Adjusts the randomness of text generation; higher values produce more diverse outputs. Default value is 1.0.")
    parser.add_argument('--top_k', type=int, default=9, help='The number of highest probability tokens to consider for top-k sampling. Defaults to 9.')
    parser.add_argument('--top_p', type=float, default=0.9, help='The cumulative probability threshold (0.0 - 1.0) for top-p (nucleus) sampling. It defines the minimum subset of tokens to consider for random sampling. Defaults to 0.9.')
    parser.add_argument('--min_atoms', type=int, default=None, help='Minimum number of non-H atoms allowed for generation.')
    parser.add_argument('--max_atoms', type=int, default=35, help='Maximum number of non-H atoms allowed for generation. Default value is 35.')
    parser.add_argument('--no_limit', action='store_true', default=False, help='Disable the default max atoms limit.')


    args = parser.parse_args()
    protein_seq = args.pro_seq
    fasta_file = args.fasta
    ligand_prompt = args.ligand_prompt
    directly_gen = args.empty_input
    num_generated = args.number
    device = args.device
    output_path = args.output
    batch_generated_size = args.batch_size
    temperature_value = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    min_atoms = args.min_atoms
    max_atoms = args.max_atoms

    if args.no_limit:
        max_atoms = None
    
    if (args.min_atoms is not None) and (args.max_atoms is not None) and (args.min_atoms > args.max_atoms):
        raise ValueError("Error: min_atoms cannot be greater than max_atoms.")
    
    if args.ligand_prompt:
        args.max_atoms = None
        args.min_atoms = None
        print("Note: --ligand_prompt is specified. --max_atoms and --min_atoms settings will be ignored.")
    
    logging.basicConfig(level=logging.CRITICAL)
    openbabel.obErrorLog.StopLogging()
    os.makedirs(output_path, exist_ok=True)
    # Check if the input is either a protein amino acid sequence or a FASTA file, but not both
    if directly_gen:
        print("Now in directly generate mode.")
        prompt = "<|startoftext|><P>"
        print(prompt)
    else:
        if (not protein_seq) and (not fasta_file):
            print("Error: Input is empty.")
            sys.exit(1)
        if protein_seq and fasta_file:
            print("Error: The input should be either a protein amino acid sequence or a FASTA file, but not both.")
            sys.exit(1)
        if fasta_file:
            protein_seq = read_fasta_file(fasta_file)
        # Generate a prompt for the model
        p_prompt = "<|startoftext|><P>" + protein_seq + "<L>"
        l_prompt = "" + ligand_prompt
        prompt = p_prompt + l_prompt
        print(prompt)


    # # Load the tokenizer and the model
    # tokenizer = AutoTokenizer.from_pretrained('liyuesen/druggpt')
    # model = GPT2LMHeadModel.from_pretrained("liyuesen/druggpt")
    # Load the tokenizer and the model from local directory
    # Load the tokenizer and the model from local directory
    # Load the tokenizer and the model from local directory
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import os

    # 假设 local_model_path 是你的本地模型文件夹路径
    local_model_path = "../druggpt_bin"

    # 确保路径存在
    assert os.path.isdir(local_model_path), f"路径 {local_model_path} 不存在"

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    model.eval()
    device = torch.device(device)
    model.to(device)

    # Create a LigandPostprocessor object
    ligand_post_processor = LigandPostprocessor(output_path)

    # Generate molecules
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    batch_number = 0

    directly_gen_protein_list = []
    directly_gen_ligand_list = []
    

    attention_mask = generated.ne(tokenizer.pad_token_id).float()
    while len(ligand_post_processor.hash_ligand_mapping) < num_generated:
        if len(ligand_post_processor.hash_ligand_mapping) < num_generated:
            print(
                f"Stopped early: only {len(ligand_post_processor.hash_ligand_mapping)}/{num_generated} "
                f"valid molecules after {batch_number} batches."
            )
        generate_ligand_list = []
        batch_number += 1
        print(f"=====Batch {batch_number}=====")
        print("Generating ligand SMILES ...")
        sample_outputs = model.generate(
            generated,
            do_sample=True,
            top_k=top_k,
            max_length=1024,
            top_p=top_p,
            temperature=temperature_value,
            num_return_sequences=batch_generated_size, 
            attention_mask=attention_mask,
            pad_token_id = tokenizer.eos_token_id
        )

        for sample_output in sample_outputs:
            text = tokenizer.decode(sample_output, skip_special_tokens=True)

            if '<L>' not in text:
                continue

            tail = text.split('<L>', 1)[1].strip()

            # 只取第一行，避免把后面脏内容全吞进去
            tail = tail.splitlines()[0].strip()

            # 如果后面混入空格分隔的垃圾，只取第一个 token
            tail = tail.split()[0].strip()

            if tail:
                generate_ligand_list.append(tail)


        torch.cuda.empty_cache()
        ligand_post_processor.to_sdf(generate_ligand_list)
        ligand_post_processor.delete_empty_files()
        ligand_post_processor.check_sdf()

        before_cnt = len(ligand_post_processor.hash_ligand_mapping)

        ligand_post_processor.to_sdf(generate_ligand_list)
        ligand_post_processor.delete_empty_files()
        ligand_post_processor.check_sdf()

        after_cnt = len(ligand_post_processor.hash_ligand_mapping)
        print(
            f"[Batch {batch_number}] candidates={len(generate_ligand_list)} "
            f"valid_total={after_cnt}/{num_generated} "
            f"new_valid={after_cnt - before_cnt}"
        )


    if directly_gen:
        arr = np.array([directly_gen_protein_list, directly_gen_ligand_list])
        processed_ligand_list = ligand_post_processor.hash_ligand_mapping.values()
        with open(os.path.join(output_path, 'generate_directly.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            for index in range(arr.shape[1]):
                protein, ligand = arr[0, index], arr[1, index]
                if ligand in processed_ligand_list:
                    writer.writerow([protein, ligand])

    print("Saving mapping file ...")
    ligand_post_processor.save_mapping()
    print(f"{len(ligand_post_processor.hash_ligand_mapping)} molecules successfully generated!")

    print("Ligand Energy Minimization")
    result = subprocess.run(['python', 'druggpt_min_multi.py', '-d', output_path])
