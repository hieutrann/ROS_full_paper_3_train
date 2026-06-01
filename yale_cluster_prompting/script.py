import os
import json
import re
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import ast
import torch
import numpy as np
import random
from argparse import ArgumentParser

import nltk
from nltk.tokenize import PunktSentenceTokenizer

from utils import remove_json_output
from create_train_test_pool import create_sentence_pool
# %run utils.py
# %run create_train_test_pool
random_state = 42

parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model_folder_org", type=str, required=True)
parser.add_argument("--model_folder", type=str, required=True)
args = parser.parse_args()

model = args.model
model_folder_org = args.model_folder_org
model_folder = args.model_folder

if os.path.exists(f"{model_folder_org}") == False:
    os.makedirs(f"{model_folder_org}")
    
if os.path.exists(f"{model_folder}") == False:
    os.makedirs(f"{model_folder}")


"""TRAIN TEST SPLIT"""

from sklearn.model_selection import train_test_split

train_notes, test_notes = train_test_split(
    os.listdir("notes"),
    test_size=0.9,
    random_state=42,
    shuffle=True
)
print(len(train_notes), len(test_notes))

"""Create training & testing df set"""

from sentence_transformers import SentenceTransformer
model_enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
from sklearn.cluster import KMeans

read_train_notes = []
for sample in train_notes:
    with open(f"notes/{sample}", 'r') as file:
        text = file.read()
        read_train_notes.append((sample[:-4],text))

read_test_notes = []
for sample in test_notes:
    with open(f"notes/{sample}", 'r') as file:
        text = file.read()
        read_test_notes.append((sample[:-4],text))

# Build embeddings for the training notes
df_train = pd.DataFrame({'note': read_train_notes})
df_train['filename'] = df_train['note'].str[0]
df_train['text'] = df_train['note'].str[1]
train_embeddings = model_enc.encode(df_train['text'].tolist(), convert_to_tensor=True)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
labels = kmeans.fit_predict(train_embeddings.to('cpu').numpy())
df_train['cluster'] = labels

# Build test embeddings and assign clusters based on closest centroid

df_test_temp = pd.DataFrame({'note': read_test_notes})
df_test_temp['filename'] = df_test_temp['note'].str[0]
df_test_temp['text'] = df_test_temp['note'].str[1]


test_embeddings = model_enc.encode(df_test_temp['text'].tolist(), convert_to_tensor=True)

cluster_centroids = {}
for cluster_id in sorted(df_train['cluster'].unique()):
    cluster_mask = torch.tensor(df_train['cluster'].to_numpy() == cluster_id)
    cluster_centroids[cluster_id] = train_embeddings[cluster_mask].mean(dim=0)

centroid_ids = list(cluster_centroids.keys())
centroid_matrix = torch.stack([cluster_centroids[cluster_id] for cluster_id in centroid_ids])
distances = torch.cdist(test_embeddings, centroid_matrix)
closest_centroid_idx = distances.argmin(dim=1).cpu().numpy()

df_test = df_test_temp.copy()
df_test['cluster'] = [centroid_ids[idx] for idx in closest_centroid_idx]

sentences_pool, labels_pool_df = create_sentence_pool(df_train)

"""Some functions when prompting"""
def few_shots_generation(labels_pool_df, sentences_pool, cluster):
    sample_sizes = {'PROBLEM': 3, 'TEST': 2, 'DRUG': 1, 'TREATMENT':1}
    stratified_sample = labels_pool_df[labels_pool_df['cluster']==cluster].groupby("label", group_keys=True) \
                        .apply(lambda x: x.sample(n=sample_sizes.get(x.name), random_state=42)).reset_index()
                
    stratified_sample_dict = stratified_sample.to_dict("records")
    # print(stratified_sample_dict)
    matches = []
    for d1 in stratified_sample_dict:
        matched_sentence = None
        for d2 in sentences_pool:
            if (
                d1['sample'] == d2['sample'] and
                d1['start'] >= d2['start'] and
                d1['end'] <= d2['end']
            ):
                matched_sentence = d2['sentence']
                break  # stop after first match
        matches.append({**d1, 'sentence': matched_sentence})

    return matches

def return_list_of_sentences(text):
    tokenizer = PunktSentenceTokenizer()
    sentences = list(tokenizer.span_tokenize(text))
    # sentence_spans = [{'start': start, 'end': end, 'sentence': text[start:end], 'sample':sample}
    #                 for start, end in sentences]
    sentences_list = [text[start:end] for start, end in sentences]

    return sentences_list


""" running time measurement """
import time

start_time = time.time()

# code block here


"""PROMPTING AND INFERENCE"""
list_ran = []
for idx, t in enumerate(sorted(zip(df_test['filename'], df_test['cluster']))):
    # if t[0] in ['sample_2755','sample_4_20210112211655','sample_65_20210604143822_Booma','sample_69_20210112211703','sample_82_20210112211705']:
        sample = t[0]
        cluster = t[1]
        print(f"processing {sample}:")
        """PROMPT AND MODEL"""
        prompt = ChatPromptTemplate([
            ("system", """You are a clinical Named Entity Recognition (NER) assistant.

             Extract all medical entities from the clinical text and classify each entity as one of:
             - PROBLEM: diseases, symptoms, disorders, findings
             - TEST: laboratory tests, diagnostics, exams, medical imaging
             - DRUG: medications, antibiotics, vaccines, prescriptions
             - TREATMENT: procedures, interventions, and substances given to a patient for treating a problem

             Rules:
             1. Return valid JSON only.
             2. Use double quotes only.
             3. Do not include explanations or any text outside the JSON.
             4. Preserve the exact entity surface form from the source text.
             5. Keep repeated mentions as separate entries.
             6. If no entities are found, return [].

             Output format:
             [
               {{"entity": "label"}}
             ]
             """),
            ("human", "Use the following examples as guidance for the output format and labeling decisions.\n\n{few_shots}\n\nNow extract entities from this clinical text.\n\nText:\n{sent}")
        ])
        
        matches = few_shots_generation(labels_pool_df, sentences_pool, cluster=cluster)
        
        few_shots = []
        for idx, i in enumerate(matches):
            test_text = f"Example {idx+1}\n"
            test_text += f"Text:\n{i.get('sentence')}\n\nOutput:\n[\n  {{\"{i.get('entity')}\": \"{i.get('label')}\"}}\n]\n" 
            few_shots.append(test_text)

        few_shots_text = "\n\n".join(few_shots)

        #Read and chunk text
        with open(f'notes/{sample}.txt', 'r') as file:
            text = file.read()

        # chunks = efficient_chunk_text(text)
        chunks = return_list_of_sentences(text)

        llm=ChatOllama(
            model=model, temperature = 0.0, num_ctx = 8192, repeat_penalty = 1, seed=42, reasoning=False, top_k = 1, top_p = 1.0
        )

        output = []
        for sent in chunks:
        #Chaining and inference
            chain = prompt|llm
            output.append(chain.invoke({"few_shots":few_shots_text, "sent": {sent}}))

        # Process and save output
        output_json = []         
        for i in range(len(output)):
            print(f"processing chunk {i}:")
            # print(output[i])
            output_json.append(remove_json_output(output[i].content))

        # Final processing to ensure valid JSON and handle any malformed outputs
        preds_json = []
        def parse_mixed_list(s):
            try:
                return ast.literal_eval(s) 
            
            except:
                s = json.dumps(s)
                return s


        for item in output_json:
            parsed = parse_mixed_list(item)
            preds_json.extend(parsed)


        with open(f"{model_folder_org}/{sample}", 'w') as f:
            f.write(str(preds_json))

        list_ran.append(sample)


end_time = time.time()
print(f"Running time of {model} is {end_time - start_time:.2f} seconds")
import csv
csv_file = "runtime_log.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow(["model", "running_time_seconds"])

    writer.writerow([model, {round(end_time - start_time, 2)}])

"""POST-PROCESSING AND CLEANING"""

LABELS = {"PROBLEM", "TEST", "DRUG", "TREATMENT"}

def strip_code_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text)
        text = re.sub(r"\\s*```$", "", text)
    return text.strip()

def extract_json_like_block(text):
    text = strip_code_fences(text)
    for left, right in [("[", "]"), ("{", "}")]:
        start = text.find(left)
        end = text.rfind(right)
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
    return text

def parse_messy_json(text):
    text = strip_code_fences(text).strip()
    if not text:
        return []

    # Handle files saved like: ['[ ... ]']
    for parser in (ast.literal_eval, json.loads):
        try:
            obj = parser(text)
            if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], str):
                text = obj[0].strip()
                break
        except Exception:
            pass

    text = extract_json_like_block(text)

    # Normal valid JSON / Python-literal path
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass

    # Existing per-object fallback
    matches = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    parsed = []
    for match in matches:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed.append(parser(match))
                break
            except Exception:
                continue
    if parsed:
        return parsed

    # New salvage path for malformed items like:
    # {"entity": "intestinal illness": "PROBLEM"}
    malformed_matches = re.findall(
        r'\{\s*"\s*entity"\s*:\s*"((?:\\.|[^"\\])*)"\s*:\s*"(PROBLEM|TEST|DRUG|TREATMENT)"\s*\}',
        text,
        flags=re.DOTALL,
    )

    cleaned = []
    seen = set()
    for entity, label in malformed_matches:
        entity = entity.strip()
        key = (entity, label)
        if entity and key not in seen:
            seen.add(key)
            cleaned.append({entity: label})

    return cleaned


# def normalize_entities(obj):
#     if obj is None:
#         return []

#     if isinstance(obj, str):
#         obj = obj.strip()
#         if not obj or obj == "[]":
#             return []
#         return normalize_entities(parse_messy_json(obj))

#     if isinstance(obj, dict):
#         cleaned = []
#         for phrase, label in obj.items():
#             if isinstance(phrase, str) and isinstance(label, str) and label in LABELS:
#                 cleaned.append({phrase.strip(): label})
#         return cleaned

#     if isinstance(obj, (list, tuple)):
#         cleaned = []
#         for item in obj:
#             cleaned.extend(normalize_entities(item))
#         return cleaned

#     return []

def normalize_entities(obj):
    if obj is None:
        return []

    if isinstance(obj, str):
        obj = obj.strip()
        if not obj or obj == "[]":
            return []
        return normalize_entities(parse_messy_json(obj))

    if isinstance(obj, dict):
        cleaned = []

        if (
            "entity" in obj
            and "label" in obj
            and isinstance(obj["entity"], str)
            and isinstance(obj["label"], str)
            and obj["label"] in LABELS
        ):
            return [{obj["entity"].strip(): obj["label"]}]

        for phrase, label in obj.items():
            if isinstance(phrase, str) and isinstance(label, str) and label in LABELS:
                cleaned.append({phrase.strip(): label})
        return cleaned

    if isinstance(obj, (list, tuple)):
        cleaned = []
        for item in obj:
            cleaned.extend(normalize_entities(item))
        return cleaned

    return []


os.makedirs(model_folder, exist_ok=True)

written = 0
for sample in sorted(os.listdir(model_folder_org)):
    with open(f"{model_folder_org}/{sample}", "r") as f:
        raw_output = f.read()

    parsed_output = parse_messy_json(raw_output)
    cleaned_output = normalize_entities(parsed_output)

    with open(f"{model_folder}/{sample}", "w") as f:
        json.dump(cleaned_output, f, ensure_ascii=False, indent=2)

    written += 1

print(f"Wrote {written} cleaned samples to {model_folder}")
if not cleaned_output:
    print(f"Still empty after salvage: {sample}")

