#!/usr/bin/env python
# coding: utf-8


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

trained_model_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'
trained_tokenizer_path = 'ZhangCheng/T5-Base-Fine-Tuned-for-Question-Generation'


class QuestionGeneration:

    def __init__(self, model_dir=None):
        self.model = T5ForConditionalGeneration.from_pretrained(trained_model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(trained_tokenizer_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, answer: str, context: str):
        input_text = '<answer> %s <context> %s ' % (answer, context)
        encoding = self.tokenizer.encode_plus(
            input_text,
            return_tensors='pt'
        ).to(self.device)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams = 3,
            num_return_sequences = 1
        )
        # question_list = []
        # for output in outputs:
        #     question = self.tokenizer.decode(
        #         output,
        #         skip_special_tokens=True,
        #         clean_up_tokenization_spaces=True
        #     )
        #     question_list.append(question)
        # return {'question': question_list, 'answer': answer, 'context': context}
        question = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return {'question': question, 'answer': answer, 'context': context}


if __name__ == "__main__":
    context = '''
    ZhangCheng fine-tuned T5 on SQuAD dataset for question generation.
    '''
    answer_list = ['ZhangCheng', 'SQuAD', 'question generation']

    QG = QuestionGeneration()

    for answer in answer_list:
        qa_pair = QG.generate(answer, context)
        print('question: ', qa_pair['question'])
        print('answer: ', qa_pair['answer'])

