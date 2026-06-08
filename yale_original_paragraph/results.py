import os
import json
import re
import pprint
import csv

from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from utils import count_relaxed_matches, count_exact_matches, exact_match_results, relax_match_results, sample_to_prediction_list
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_folder", type=str, required=True)
args = parser.parse_args()

model_folder = args.model_folder

# RESULTS

# if os.path.exists(f"results.csv") == True:
#     os.remove(f"results.csv")


tp_rm, fp_rm, fn_rm, od_rm = 0, 0, 0, 0
tp_em, fp_em, fn_em, od_em = 0, 0, 0, 0
for sample in os.listdir(model_folder):
    # if sample in ['sample_68', 'sample_635', 'sample_71', 'sample_324','sample_2748']:
        print(f"evaluating {sample}:")
        with open(f'label/{sample}.json', 'r') as f:
            label_dict = json.load(f)

        #Read output
        with open(f"{model_folder}/{sample}", 'r') as f:
            preds_dict = json.load(f)
        print(preds_dict)
        #Read original text
        with open(f'notes/{sample}.txt', 'r') as file:
            text = file.read()

        preds_dict_processed = []
        for ent in preds_dict:
            for phrase, label in ent.items():
        
                matches = list(re.finditer(re.escape(phrase), text,re.IGNORECASE))
                if matches:
                    for match in matches:
                        # print(f"Found '{match.group()}' at [{match.start()}, {match.end()}]")
                        preds_dict_processed.append({'entity':match.group(), 'label': label, 'start': match.start(), 'end': match.end()})
                else:
                    preds_dict_processed.append({'entity':phrase, 'label': 'OD', 'start': 0, 'end': 0})
                
        #Deduplication
        preds_dict_final = []
        seen = set()
        for d in preds_dict_processed:
            t = tuple(sorted(d.items()))
            if t not in seen:
                seen.add(t)
                preds_dict_final.append(d)
                
        ## Evaluation
        em, em_mm, em_ud, em_od = exact_match_results(label_dict, preds_dict_final, 'Output_verification', sample)
        rm, rm_mm, rm_ud, rm_od = relax_match_results(label_dict, preds_dict_final, 'Output_verification', sample)
    
        tp_rm+= rm
        fp_rm+= rm_mm
        fn_rm+= rm_ud
        od_rm+= rm_od

            
        tp_em+= em
        fp_em+= em_mm
        fn_em+= em_ud
        od_em+= em_od

rm_precision = tp_rm/(tp_rm + fp_rm + od_rm)
rm_recall = tp_rm/(tp_rm + fn_rm)

em_precision = tp_em/(tp_em + fp_em + od_em)
em_recall = tp_em/(tp_em + fn_em)

f1_em = round(2*em_precision*em_recall/(em_precision + em_recall),4)
f1_rm = round(2*rm_precision*rm_recall/(rm_precision + rm_recall),4)

print(f"SUMMARY EXACT MATCH: TP {tp_em}, FP {fp_em}, UD {fn_em}, OD {od_em}")
print(f"MICRO AVERAGE SCORE EXACT MATCH: Precision: {em_precision}, Recall: {em_recall}, F1 score: {f1_em}")

print(f"SUMMARY RELAX MATCH: TP {tp_rm}, FP {fp_rm}, UD {fn_rm}, OD {od_rm}")
print(f"MICRO AVERAGE SCORE RELAX MATCH: Precision: {rm_precision}, Recall: {rm_recall}, F1 score: {f1_rm}")

with open('results.csv', "a", newline="") as f:
    writer = csv.writer(f)
    
    if not os.path.isfile('results.csv'):    
        writer.writerow(["model", "f1_em", "f1_rm"])

    writer.writerow([model_folder, f1_em, f1_rm])

        