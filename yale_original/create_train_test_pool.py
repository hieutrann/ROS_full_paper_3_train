import os
import torch
import pandas as pd
random_state = 42
import nltk
from nltk.tokenize import PunktSentenceTokenizer

def create_train_test_pool():
    ## Read and embed notes
    notes = []
    for sample in os.listdir('notes'):
        with open(f"notes/{sample[:-4]}.txt", 'r') as file:
            text = file.read()
            notes.append((sample[:-4],text))


    from sentence_transformers import SentenceTransformer
    vecs = []
    model_enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for i in range(len(notes)):
        vec = model_enc.encode(notes[i][1])
        vecs.append(torch.tensor(vec))

    embeddings = torch.stack(vecs)

    #K-means to cluster the notes
    from sklearn.cluster import KMeans
    import numpy as np

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)

    df = pd.DataFrame({'note': notes, 'cluster': labels})
    df['filename'] = df['note'].str[0]
    df['text'] = df['note'].str[1]

    df = df.drop(columns='note')
    df['cluster'].value_counts()

    df_0 = df[df['cluster']==0]
    df_1 = df[df['cluster']==1]
    df_2 = df[df['cluster']==2]
    df_3 = df[df['cluster']==3]
    df_4 = df[df['cluster']==4]

    df_0_train_1 = df_0.sample(n=1, random_state=random_state)
    df_1_train_1 = df_1.sample(n=3, random_state=random_state)
    df_2_train_1 = df_2.sample(n=1, random_state=random_state)
    df_3_train_1 = df_3.sample(n=1, random_state=random_state)
    df_4_train_1 = df_4.sample(n=1, random_state=random_state)

    train_df = pd.concat([df_0_train_1, df_1_train_1,df_2_train_1,df_3_train_1,df_4_train_1])
    test_df = df.drop(index=train_df.index)

    # #Print
    # print(train_df)
    # print(test_df)

    return train_df, test_df


# Create sentence pool and labels pool
def create_sentence_pool(train_df):
    ## Create sentence pool from training notes
    tokenizer = PunktSentenceTokenizer()
    sentences_pool_temp = []
    for sample in train_df['filename'].to_list():
        with open(f'notes/{sample}.txt', 'r') as file:
            text = file.read()
        sentences = list(tokenizer.span_tokenize(text))

    # Convert to list of dicts for convenience
        sentence_spans = [{'start': start, 'end': end, 'sentence': text[start:end], 'sample':sample}
                        for start, end in sentences]
        sentences_pool_temp.append(sentence_spans)

    sentences_pool = [s for sentence in sentences_pool_temp for s in sentence]

    labels_pool_temp = []
    for t in list(zip(train_df['filename'],train_df['cluster'])):
        sample = t[0]
        cluster = t[1]
        df = pd.read_csv(f"annotation/{sample}.ann", sep='\t', header = None, names=['tag','annotate','entity'])
        df = df[~(df["tag"].str.startswith("R")) & (df['annotate'].str.startswith("problem")) | (df['annotate'].str.startswith("drug")) | (df['annotate'].str.startswith("test")) | (df['annotate'].str.startswith("treatment"))]
        df['sample'] = sample
        df['cluster'] = cluster
        df[['label', 'start', 'end']] = df['annotate'].str.split(" ", expand=True)
        df['label'] = df['label'].str.upper()
        df = df.drop(columns=['tag', 'annotate'])
        labels_pool_temp.append(df)

        labels_pool_df = pd.concat(labels_pool_temp, ignore_index=True)
        labels_pool_df['start'] = labels_pool_df['start'].astype('int')
        labels_pool_df['end'] = labels_pool_df['end'].astype('int')

    return sentences_pool, labels_pool_df

## Return labels with their coresponding sentences
def return_label_sentence(labels_pool_df, sentences_pool):
    list_labels_dict = labels_pool_df.to_dict("records")
    for d in list_labels_dict:
        d.pop("cluster")

    matches = []
    for d1 in list_labels_dict:
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