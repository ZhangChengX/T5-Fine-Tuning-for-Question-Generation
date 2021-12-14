#!/usr/bin/env python
# coding: utf-8

# Download, preprocess, and save dataset

from tqdm import tqdm
# from tqdm.notebook import tqdm
from datasets import load_dataset
from sklearn.utils import shuffle
import pandas as pd
import os


print('Downloading SQuAD dataset...')
train_dataset = load_dataset("squad", split='train')
valid_dataset = load_dataset("squad", split='validation')
print('train: ', len(train_dataset))
print('validation: ', len(valid_dataset))

pd.set_option('display.max_colwidth', None) # use None to not limit the column width
df_train = pd.DataFrame(columns = ['context', 'answer', 'question'])
df_validation = pd.DataFrame(columns = ['context', 'answer', 'question'])

print('Loading df_train...')
num_of_long_answer = 0
num_of_short_answer = 0
for index, value in tqdm(enumerate(train_dataset)):
    context = value['context']
    question = value['question']
    answer = value['answers']['text'][0]
    number_of_words = len(answer.split())
    if number_of_words >= 8:
        num_of_long_answer = num_of_long_answer + 1
        continue
    else:
        df_train.loc[num_of_short_answer] = [context] + [answer] + [question]
        num_of_short_answer = num_of_short_answer + 1
print('Long answer train dataset: ', num_of_long_answer)
print('Short answer train dataset: ', num_of_short_answer)

print('Loading df_validation...')
num_of_long_answer = 0
num_of_short_answer = 0
for index, value in tqdm(enumerate(valid_dataset)):
    context = value['context']
    question = value['question']
    answer = value['answers']['text'][0]
    number_of_words = len(answer.split())
    if number_of_words >= 8:
        num_of_long_answer = num_of_long_answer + 1
        continue
    else:
        df_train.loc[num_of_short_answer] = [context] + [answer] + [question]
        num_of_short_answer = num_of_short_answer + 1
print('Long answer validation dataset: ', num_of_long_answer)
print('Short answer validation dataset: ', num_of_short_answer)

print('Shuffling DataFrame...')
df_train = shuffle(df_train)
df_validation = shuffle(df_validation)

# print('df_train.shape')
# print(df_train.shape)
# print('df_validation.shape')
# print(df_validation.shape)
# print('df_train.head():')
# print(df_train.head())
# print('df_validation.head():')
# print(df_validation.head())

print('Saving dataset as csv...')
dataset_save_path = 'datasets/'
train_file_path = dataset_save_path + 'squad_train.csv'
validation_file_path = dataset_save_path + 'squad_validation.csv'
if not os.path.exists(dataset_save_path):
    os.makedirs(dataset_save_path)
df_train.to_csv(train_file_path, index=False)
df_validation.to_csv(validation_file_path, index=False)
print('All done.')



