import os
import json
import re
import pandas as pd
import nltk
import tiktoken
nltk.download('punkt_tab')


"""
REMOVE JSON OUTPUT FROM LLM
"""
def remove_json_output(pred_text):
    match = re.search(r"```json\s*(.*?)\s*```", pred_text, re.DOTALL)
    if match:
        preds_text_extract = match.group(1).strip()
        try:            
            preds_text_extract = json.dumps(preds_text_extract)
            preds_json = json.loads(preds_text_extract)
        except:
            pred_text = pred_text.replace("'", '"').replace("'", '"')
            match = re.search(r"```json\s*(.*?)\s*```", pred_text, re.DOTALL)
            preds_text_extract = match.group(1).strip()
            preds_json = json.loads(preds_text_extract)

    else:
        print("No JSON found.")
        preds_json = pred_text
    return preds_json

#fix for llama3.1, sample_227 and sample_64: malformed json /dict {"a", "b","c", "TEST"}

def malformed_json_fix(text):
    return re.sub(r'\{([^{}:]*),\s*"([^"]+)":\s*"([^"]+)"\}',
                            lambda m: '{' + '"' + re.sub(r'"\s*,\s*"', ', ', m.group(1).strip('" ')) + f', {m.group(2)}": "{m.group(3)}' + '"}',
                            text)
"""
Efficiently chunk text by sentence with token-level control and overlap.
"""
def efficient_chunk_text(text, model_name="gpt-4o-mini", max_tokens=300, overlap=100):

    enc = tiktoken.encoding_for_model(model_name)
    sentences = nltk.sent_tokenize(text)

    chunks, current_chunk, tokens_so_far = [], [], 0

    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))
        
        if tokens_so_far + sentence_tokens > max_tokens:
            # finalize chunk
            chunks.append(" ".join(current_chunk))
            # add overlap (keep last few sentences)
            overlap_tokens, overlap_chunk = 0, []
            for s in reversed(current_chunk):
                stoks = len(enc.encode(s))
                if overlap_tokens + stoks > overlap: break
                overlap_chunk.insert(0, s)
                overlap_tokens += stoks
            current_chunk = overlap_chunk
            tokens_so_far = overlap_tokens
        
        current_chunk.append(sentence)
        tokens_so_far += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


"""
COUNT RELAX MATCHES
"""
def count_relaxed_matches(label_dict, pred_dict):
    used_preds = set()
    relax_match = []
    false_positive = []
    gold_labels_used = []
    for gold in label_dict:
        for i, pred in enumerate(pred_dict):
            if i in used_preds:
                continue
            if gold["label"] == pred["label"]:
                if min(gold["end"], pred["end"]) >= max(gold["start"], pred["start"]):
                    used_preds.add(i)
                    relax_match.append(pred)
                    gold_labels_used.append(gold)
            elif gold["label"] != pred["label"]:
                if min(gold["end"], pred["end"]) >= max(gold["start"], pred["start"]):
                    used_preds.add(i)
                    false_positive.append(pred)
                    gold_labels_used.append(gold)
    return relax_match, false_positive, gold_labels_used

def relax_match_results(label_dict, preds_dict_final, model_folder, sample):
    
    relax_match, false_positive, gold_label_used = count_relaxed_matches(label_dict,preds_dict_final)
        
    over_detection = [e for e in preds_dict_final if e not in relax_match and e not in false_positive]
    print(over_detection)
    print(false_positive)
    under_detection = []
    for gold in label_dict:
                if gold in relax_match or gold in gold_label_used:
                    continue
                else: under_detection.append(gold)

    # Metric:
    if (len(relax_match) + len(false_positive) + len(over_detection)) == 0:
        precision = 0
    else:
        precision = len(relax_match) / (len(relax_match) + len(false_positive) + len(over_detection))
    
    if (len(relax_match) + len(under_detection)) == 0:
        recall = 0
    else:
        recall = len(relax_match) / (len(relax_match) + len(under_detection))

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    print(f"RM {sample} results:")
    print(len(relax_match),len(false_positive),len(under_detection),len(over_detection))
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1_score:.4f}\n\n")

    with open(f"results/{model_folder}_rm.csv", 'a') as f:
        f.writelines(f'{sample},{len(relax_match)},{len(false_positive)}, {len(under_detection)},{len(over_detection)},{precision:.4f},{recall:.4f},{f1_score:.4f}')
        f.writelines('\n')
    
    return len(relax_match),len(false_positive), len(under_detection), len(over_detection)

"""
COUNT EXACT MATCHES
"""
def count_exact_matches(label_dict, pred_dict):
    used_preds = set()
    exact_match = []
    false_positive = []
    gold_labels_used = []
    for gold in label_dict:
        for i, pred in enumerate(pred_dict):
            if i in used_preds:
                continue
            if gold["label"] == pred["label"]:
                if gold["end"] == pred["end"] and gold["start"] == pred["start"]:
                    used_preds.add(i)
                    exact_match.append(pred)
                    gold_labels_used.append(gold)
            elif gold["label"] != pred["label"]:
                if gold["end"] == pred["end"] and gold["start"] == pred["start"]:
                    used_preds.add(i)
                    false_positive.append(pred)
                    gold_labels_used.append(gold)
    return exact_match, false_positive, gold_labels_used



def exact_match_results(label_dict, preds_dict_final, model_folder, sample):
    exact_match, false_positive, gold_label_used = count_exact_matches(label_dict,preds_dict_final)
    # print(exact_match)
    over_detection = [e for e in preds_dict_final if e not in exact_match and e not in false_positive]
    print(over_detection)
    print(false_positive)
    under_detection = []
    for gold in label_dict:
            if gold in exact_match or gold in gold_label_used:
                continue
            else: under_detection.append(gold)


    # Metric:
    if (len(exact_match) + len(false_positive) + len(over_detection)) == 0:
        precision = 0
    else:
        precision = len(exact_match) / (len(exact_match) + len(false_positive) + len(over_detection))
    
    if (len(exact_match) + len(under_detection)) == 0:
        recall = 0
    else:
        recall = len(exact_match) / (len(exact_match) + len(under_detection))

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    print(f"EM {sample} results:")
    print(len(exact_match),len(false_positive),len(under_detection),len(over_detection))
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1_score:.4f}\n\n")
        
    with open(f"results/{model_folder}_em.csv", 'a') as f:
        f.writelines(f'{sample},{len(exact_match)},{len(false_positive)}, {len(under_detection)},{len(over_detection)},{precision:.4f},{recall:.4f},{f1_score:.4f}')
        f.writelines('\n')

    return len(exact_match),len(false_positive), len(under_detection), len(over_detection)


"""Read original output, match with original text and issues final json prediction
"""
def sample_to_prediction_list(model_number, sample):
    with open(f"{model_number}/{sample}", 'r') as f:
        content = f.read()
        # Remove opening & closing fence like ```json or ```JSON
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        preds_json = json.loads(content)

    #Read original text
    with open(f'notes/{sample}.txt', 'r') as file:
        text = file.read()

    preds_dict = []
    for ent in preds_json:
        for phrase, label in ent.items():
    
            matches = list(re.finditer(re.escape(phrase), text, re.IGNORECASE))
            if matches:
                for match in matches:
                    # print(f"Found '{match.group()}' at [{match.start()}, {match.end()}]")
                    preds_dict.append({'entity':match.group(), 'label': label, 'start': match.start(), 'end': match.end()})
            else:
                preds_dict.append({'entity':phrase, 'label': 'OD', 'start': 0, 'end': 0})
            
    #Deduplication
    preds_dict_final = []
    seen = set()
    for d in preds_dict:
        t = tuple(sorted(d.items()))
        if t not in seen:
            seen.add(t)
            preds_dict_final.append(d)
    return preds_dict_final


def output_individual_sample(sample):
    print(f"processing {sample}:")
    df = pd.read_csv(f"annotation/{sample}.ann", sep='\t', header = None, names=['tag','annotate','entity'])
    df = df[~(df["tag"].str.startswith("R")) & (df['annotate'].str.startswith("problem")) | (df['annotate'].str.startswith("drug")) | (df['annotate'].str.startswith("test"))]

    # Split label, start, end
    split_cols = df['annotate'].str.split(' ', expand=True)
    split_cols.columns = ['label','start','end']

    # Concatenate with original df
    df = pd.concat([df, split_cols], axis=1)
    label_df = df[['entity','label','start','end']]
    label_df_cp = label_df.copy()
    label_df_cp['label'] = label_df_cp['label'].str.upper()
    label_df_cp['start'] = label_df_cp['start'].astype(int)
    label_df_cp['end'] = label_df_cp['end'].astype(int)

    label_dict = []
    for _, row in label_df_cp.iterrows():
        label_dict.append(dict(row))


    if os.path.exists(f"{model_folder}") == False:
        os.makedirs(f"{model_folder}")

    with open(f'notes/{sample}.txt', 'r') as file:
        text = file.read()

    chunks = efficient_chunk_text(text)
    # for i, c in enumerate(chunks, 1):
    #     print(f"Chunk {i} ({len(c.split())} words):\n{c}\n")

    label_sample_df = pd.DataFrame(label_dict).drop(columns=['start','end'])

    # Stratified sample: 3:2:1 ratio for PROBLEM TEST DRUG from each group, based on number of labels
    if label_sample_df[label_sample_df['label']=='DRUG'].shape[0] >= 1 and label_sample_df[label_sample_df['label']=='TEST'].shape[0] >= 2:
        sample_sizes = {'PROBLEM': 3, 'TEST': 2, 'DRUG': 1}
    else:
        sample_sizes = {'PROBLEM': 5, 'TEST': 1}
        
    stratified_sample = label_sample_df.groupby("label", group_keys=False).apply(lambda x: x.sample(n=sample_sizes.get(x.name,0), random_state=42))
    stratified_sample_dict = stratified_sample.to_dict("records")
    shots = [{list(d.values())[0]: list(d.values())[1]} for d in stratified_sample_dict]

    # Insert first chunk of text and 5 random example into few shots
    few_shots = f"Example:\nText:\n {chunks[0]}\nOutput: ```json{shots}"


    """PROMPT AND MODEL"""
    prompt = ChatPromptTemplate([
        ("system", "You are an expert in biomedical and clinical domain, specialized in Named Entity Recognition (NER) from clincal notes"
        "Your task is to indentify and extract entities from clinical notes and classify each entity into 'PROBLEM', 'TEST' or 'DRUG' category"
        "Output must be valid JSON block. Do NOT include explanations, comments, or text outside the JSON. "
        "One entity may appear multiple times in the output, keep all of them."
        ""
        ),
        ("human", "{few_shots}\n\nNow extract entities from following text:\n{sent}")
    ])


    llm=ChatOllama(
        model="gemma3:27b", temperature = 0.0, num_ctx = 8192, repeat_penalty = 1.1, seed=42
    )

    output = []
    for sent in chunks:
    #Chaining and inference
        chain = prompt|llm
        output.append(chain.invoke({"few_shots":few_shots, "sent": sent}))

    output_json = []
    for i in range(len(output)):
        print(f"processing chunk {i}:")
        if i == 4:
            continue
        output_json.append(remove_json_output(output[i].content))

    preds_json = [ent for entities_list in output_json for ent in entities_list]

    with open(f"{model_folder}/{sample}", 'w') as f:
        json.dump(preds_json, f, indent=2)

