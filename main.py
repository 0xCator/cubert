from transformers import TFAutoModel 
import torch
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
import threading
from concurrent.futures import ThreadPoolExecutor

model_path = "claudios/cubert-20210711-Java-1024"
model = TFAutoModel.from_pretrained(model_path)

import pandas as pd

#parts not working: beautified2-part1, b3-p3-p2-p1, b5-p

train_df = pd.read_csv("train_beautified5.csv")
part_1 = train_df[:int(len(train_df)/4)]
part_2 = train_df[int(len(train_df)/4):int(len(train_df)/2)]
part_3 = train_df[int(len(train_df)/2):int(3*len(train_df)/4)]
part_4 = train_df[int(3*len(train_df)/4):]
part_1.index = range(len(part_1))
part_2.index = range(len(part_2))
part_3.index = range(len(part_3))
part_4.index = range(len(part_4))
train_df = part_2


from cubert.full_cubert_tokenizer import FullCuBertTokenizer
from cubert import java_tokenizer

fullTokenizer = FullCuBertTokenizer(java_tokenizer.JavaTokenizer, "20210711_Java_github_java_deduplicated_vocabulary.txt")
physical_devices = tf.config.list_physical_devices('GPU')

split_lines = [line.split('\n') for line in train_df['code']]


train_df_new = pd.DataFrame(columns = ['embedding', 'godclass', 'longmethod'])

batch_size = 16
mutex = threading.Lock()  # Create a mutex lock
train_df_new = []  # Shared list for thread-safe appending

def process_batch(batch_lines):
    """Pad batch, run through model, and return summed embeddings."""
    batch_lines_padded = pad_sequences(batch_lines, padding='post', truncating='post')
    inputs_tensor = tf.convert_to_tensor(batch_lines_padded)
    outputs = model(inputs_tensor)

    # Sum embeddings efficiently
    return np.sum(outputs[0].numpy()[:, 0, :], axis=0)

def process_sample(i, sample):
    """Tokenize, process batch, and compute sample embeddings."""
    sample_final = np.zeros(1024)
    batch_lines = []

    for line in sample:
        inputs = fullTokenizer.convert_tokens_to_ids(fullTokenizer.tokenize(line))
        if len(inputs) == 0:
            continue
        batch_lines.append(inputs)

        # Process batch when full
        if len(batch_lines) == batch_size:
            sample_final += process_batch(batch_lines)
            batch_lines = []

    # Ensure last batch is processed
    if batch_lines:
        sample_final += process_batch(batch_lines)

    result = {
        'embedding': sample_final, 
        'godclass': train_df['godclass'][i], 
        'longmethod': train_df['longmethod'][i]
    }

    # Lock while modifying shared data
    with mutex:
        train_df_new.append(result)

# Multi-threaded processing
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(lambda args: process_sample(*args), enumerate(split_lines)), total=len(split_lines)))

# Convert results to DataFrame
train_df_new = pd.DataFrame(train_df_new)

train_df_new.to_pickle("./train5-1.pkl")