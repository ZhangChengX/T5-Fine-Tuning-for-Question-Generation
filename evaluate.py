#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
from datasets import load_dataset
from datasets import load_metric
from generate import QuestionGeneration
from sentence_embeddings import SentenceEmbeddings
from nltk.tokenize import word_tokenize


if __name__ == "__main__":
    print('Downloading SQuAD dataset...')
    valid_dataset = load_dataset("squad", split='validation')

    print('Loading metrics...')
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    meteor = load_metric("meteor")
    bertscore = load_metric("bertscore")

    print('Loading QG model...')
    QG = QuestionGeneration()
    SE = SentenceEmbeddings()

    print('Generating questions...')
    i = 0
    references = []
    predictions = []
    for d in tqdm(valid_dataset):
        if i > 3: break
        i += 1
        answer = d['answers']['text'][0]
        question = d['question']
        context = d['context']
        data_id = d['id']
        references.append(question)
        print('ID: ' + data_id)
        qa_pair_list = QG.generate(answer, context)
        generated_question = SE.get_most_similar(context, qa_pair_list)
        predictions.append(generated_question['question'])

    print('Compute bleu...')
    bleu_references = [[word_tokenize(r)] for r in tqdm(references)]
    bleu_predictions = [word_tokenize(r) for r in tqdm(predictions)]
    results = bleu.compute(predictions=bleu_predictions, references=bleu_references)
    print(results)

    print('Compute rouge...')
    results = rouge.compute(predictions=predictions, references=references)
    print(results)

    print('Compute meteor...')
    results = meteor.compute(predictions=predictions, references=references)
    print(results)

    print('Compute bertscore...')
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    print('precision', sum(results['precision']) / len(results['precision']))
    print('recall', sum(results['recall']) / len(results['recall']))
    print('f1', sum(results['f1']) / len(results['f1']))



