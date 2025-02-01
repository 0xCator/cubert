from transformers import TFAutoModel 
import torch
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pickle

model_path = "claudios/cubert-20210711-Java-1024"
model = TFAutoModel.from_pretrained(model_path)



import pandas as pd

train_df = pd.read_csv("train_beautified.csv")
test_df = pd.read_csv("test_beautified.csv")


from cubert.full_cubert_tokenizer import FullCuBertTokenizer
from cubert import java_tokenizer

fullTokenizer = FullCuBertTokenizer(java_tokenizer.JavaTokenizer, "20210711_Java_github_java_deduplicated_vocabulary.txt")


physical_devices = tf.config.list_physical_devices('GPU')

split_lines = [line.split('\n') for line in train_df['code']]

train_df_new = pd.DataFrame(columns = ['embedding', 'godclass'])

from keras.preprocessing.sequence import pad_sequences

batch_size = 16 

batch_lines = []
batch_samples = []
for i, sample in enumerate(split_lines):
    print(f'{i}/{len(split_lines)}')
    sample_final = np.zeros(1024)
    
    for line in sample:
        inputs = fullTokenizer.convert_tokens_to_ids(fullTokenizer.tokenize(line))
        if len(inputs) == 0:
            continue
        batch_lines.append(inputs)
        
        if len(batch_lines) == batch_size or i == len(split_lines) - 1:
            batch_lines_padded = pad_sequences(batch_lines, padding='post', truncating='post')
            
            inputs_tensor = tf.convert_to_tensor(batch_lines_padded)
            
            outputs = model(inputs_tensor)

            for output in outputs[0]:
                sample_final += output[0].numpy()
            
            batch_lines = [] 

    train_df_new = train_df_new._append({'embedding': sample_final, 'godclass': train_df['godclass'][i]}, ignore_index=True)


train_df_new.to_pickle("./training_God_of🥒.pkl")