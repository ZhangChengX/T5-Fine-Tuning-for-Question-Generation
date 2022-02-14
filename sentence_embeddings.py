#!/usr/bin/env python
# coding: utf-8


from sentence_transformers import SentenceTransformer, util

trained_model = 'all-distilroberta-v1'
# trained_model = 'all-roberta-large-v1'


class SentenceEmbeddings:

    def __init__(self):
        self.embedder = SentenceTransformer(trained_model)

    def encode(self, text):
        return self.embedder.encode(text, convert_to_tensor=True)

    def get_most_similar(self, context:str, qa_list:list):
        context_embeddings = self.encode(context)
        top1 = {'idx': 0, 'score': float('-inf')}
        for i in range(len(qa_list)):
            qa_str = qa_list[i]['question'] + ' ' + qa_list[i]['answer']
            qa_embeddings = self.encode(qa_str)
            cos_score = util.pytorch_cos_sim(context_embeddings, qa_embeddings)
            # print(cos_score[0][0], qa_list[i])
            if cos_score[0][0] > top1['score']:
                top1['score'] = cos_score[0][0]
                top1['idx'] = i
        return qa_list[top1['idx']]


if __name__ == "__main__":
    pass



