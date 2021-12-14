#!/usr/bin/env python
# coding: utf-8

import os
import time
import copy
import torch
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm


train_file_path = 'datasets/squad_train.csv'
validation_file_path = 'datasets/squad_validation.csv'
save_model_path = 'model/'
save_tokenizer_path = 'tokenizer/'
pretrained_model = 't5-base'
# pretrained_model = 't5-large'
# pretrained_model = 'google/t5-v1_1-large'
num_workers = 0
# The batch size should pretty much be as large as possible without exceeding memory.
batch_size = 6


class QGDataset(Dataset):

    def __init__(self, tokenizer, file_path, max_len_input=512, max_len_output=128):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(file_path)
        self.max_len_input = max_len_input
        self.max_len_output = max_len_output
        self.context_column = 'context'
        self.answer_column = 'answer'
        self.question_column = 'question'
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]['input_ids'].squeeze()
        target_ids = self.targets[index]['input_ids'].squeeze()
        source_mask = self.inputs[index]['attention_mask'].squeeze()
        target_mask = self.targets[index]['attention_mask'].squeeze()
        labels = copy.deepcopy(target_ids)
        labels[labels == 0] = -100
        return {'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids, 'target_mask': target_mask, 'labels': labels}

    def _build(self):
        for idx in tqdm(range(len(self.data))):

            context, answer, target = self.data.loc[idx, self.context_column], self.data.loc[idx, self.answer_column], self.data.loc[idx, self.question_column]
            input_text = '<answer> %s <context> %s ' % (answer, context)
            target = str(target)

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_text],
                max_length=self.max_len_input,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target],
                max_length=self.max_len_output,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


class T5FineTuner(pl.LightningModule):

    def __init__(self, batch_size, num_workers, t5_model, t5_tokenizer):
        super(T5FineTuner, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = t5_model
        self.tokenizer = t5_tokenizer

    def forward(self, input_ids, attention_mask=None, decoder_attention_mask=None, lm_labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels, # decoder_input_ids included in lm_labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            decoder_input_ids=batch['target_ids'],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        loss = outputs[0]
        self.log('train_loss: ', loss)
        # print('train loss: ', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            decoder_input_ids=batch['target_ids'],
            decoder_attention_mask=batch['target_mask'],
            lm_labels=batch['labels']
        )
        loss = outputs[0]
        self.log('val_loss: ', loss)
        # print('val loss: ', loss)
        return loss

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-5, eps=1e-8)
        return optimizer


if __name__ == "__main__":

    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    pl.seed_everything(99)

    print('Loading pre-trained model...')
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    model.to(device)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )

    print('Preparing dataset...')
    train_dataset = QGDataset(tokenizer, train_file_path)
    validation_dataset = QGDataset(tokenizer, validation_file_path)

    print('train_dataset: ', len(train_dataset))
    print('validation_dataset: ', len(validation_dataset))

    print('Start fine tuning...')
    model = T5FineTuner(batch_size, num_workers, model, tokenizer)
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=1,
        auto_select_gpus=True,
        progress_bar_refresh_rate=30,
        callbacks=[EarlyStopping(monitor="val_loss")]
    )
    trainer.fit(model)

    print('Saving model...')
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    if not os.path.exists(save_tokenizer_path):
        os.makedirs(save_tokenizer_path)
    model.model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_tokenizer_path)

    end_time = time.time() - start_time
    print('Total time: %s hours' % (end_time / 60 / 60))
    print('All done.')

