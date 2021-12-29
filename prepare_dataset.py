#!/usr/bin/env python
# coding: utf-8

# Download, preprocess, and save dataset

from tqdm import tqdm
from datasets import load_dataset
from sklearn.utils import shuffle
import pandas as pd
import os


def load_squad_dataset(dataset):
    df_dataset = pd.DataFrame(columns=['context', 'question', 'answer'])
    num_of_answer = 0
    for index, value in tqdm(enumerate(dataset)):
        context = value['context']
        question = value['question']
        answer = value['answers']['text'][0]
        number_of_words = len(answer.split())
        df_dataset.loc[num_of_answer] = [context] + [question] + [answer]
        num_of_answer = num_of_answer + 1
    return df_dataset


if __name__ == "__main__":
    print('Downloading SQuAD dataset...')
    train_dataset = load_dataset("squad", split='train')
    valid_dataset = load_dataset("squad", split='validation')
    print('train: ', len(train_dataset))
    print('validation: ', len(valid_dataset))

    pd.set_option('display.max_colwidth', None)
    print('Loading df_train...')
    df_train = load_squad_dataset(train_dataset)
    print('Loading df_validation...')
    df_validation = load_squad_dataset(valid_dataset)

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
    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)
    df_train.to_csv(dataset_save_path + 'squad_train.csv', index=False)
    df_validation.to_csv(dataset_save_path + 'squad_validation.csv', index=False)
    print('All done.')

