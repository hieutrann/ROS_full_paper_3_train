import subprocess
import sys
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--script", type=str, required=True)
args = parser.parse_args()

script = args.script + ".py"

arguments = [
    # {'model': 'llama3.1:8b', 'model_folder_org': 'Output_llama3.1_8b_org', 'model_folder': 'Output_llama3.1_8b_no_sim'},
    # {'model': 'gemma4:26b', 'model_folder_org': 'Output_gemma4_26b_org', 'model_folder': 'Output_gemma4_26b_no_sim'},
    # {'model': 'gemma4:31b', 'model_folder_org': 'Output_gemma4_31b_org', 'model_folder': 'Output_gemma4_31b_no_sim'},
    {'model': 'mistral-small3.2:24b', 'model_folder_org': 'Output_mistral-small_24b_org', 'model_folder': 'Output_mistral-small_24b_no_sim'},
]

if script == 'script.py':
    for arg in arguments:
        subprocess.run([
            sys.executable,
            script,
            "--model", arg["model"],
            "--model_folder_org", arg["model_folder_org"],
            "--model_folder", arg["model_folder"],
        ], check=True)
else:
    for arg in arguments:
        subprocess.run([
            sys.executable,
            "results.py",
            "--model_folder", arg["model_folder"],
        ], check=True)