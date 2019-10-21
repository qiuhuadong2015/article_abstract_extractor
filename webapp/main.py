import collections
import os
import pickle
import numpy as np
import jieba
import gensim
from sklearn.decomposition import PCA
from nltk.cluster.util import cosine_distance
import re
from flask import Flask
from flask import render_template
from flask import request
import json

PARENT_PATH = os.path.dirname(os.path.abspath(__file__))


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    sentences = para.split("\n")
    sentences = [s for s in sentences if len(s) != 0]
    return sentences


def load_word_frequency():
    FILE = f'{PARENT_PATH}/word_frequency.pickle'

    with open(FILE, 'rb') as f:
        words_frequencys = pickle.load(f)
    
    return words_frequencys


def sentence_to_vec(sentences: list, embedding_size: int=100, a: float=1e-3):
    """
    输入句子列表，输出句子向量列表
    :param sentences: 句子列表
    :param embedding_size: 默认和gensim的word2vec参数一样，向量是100维
    :param a: SIF算法中的a系数
    :return: 100维的句子向量列表，[(100维),(100维)]
    """

    load_model = gensim.models.Word2Vec.load(f'{PARENT_PATH}/project-1-model')
    words_frequencys = load_word_frequency()

    sentence_vectors = []
    for sentence in sentences:
        sentence_vector = np.zeros(embedding_size)

        words = list(jieba.cut(sentence))
        sentence_length = len(words)

        if sentence_length == 0:
            continue

        for word in words:
            # \u3000是全角空白
            if word in [' ', '\r', '\n', '\u3000']:
                word = ','
            freq = words_frequencys[word]
            # OOV未登录词
            if freq == 0:
                a_value = 1.0
            else:
                a_value = a / (a + freq)
            sentence_vector = np.add(sentence_vector, np.multiply(a_value, load_model.wv.word_vec(word)))

        sentence_vector = np.divide(sentence_vector, sentence_length)

        sentence_vectors.append(sentence_vector)

    pca = PCA()

    pca.fit(np.array(sentence_vectors))

    u = pca.components_[0]

    u = np.multiply(u, np.transpose(u))

    if len(u) < embedding_size:
        for i in range(embedding_size - len(u)):
            u = np.append(u, 0)

    sentence_vecs = []
    for sentence_vector in sentence_vectors:
        sub = np.multiply(u, sentence_vector)
        # 转为python tuple 返回，因为要作为key，tuple可以hash
        sentence_vecs.append(tuple(np.subtract(sentence_vector, sub).tolist()))
    return sentence_vecs

def parse_text(text):

    clean_sentences = []

    sentences = cut_sent(text)

    for s in sentences:
        if s in ['', ' ', '\r', '\n', '\u3000']:
            continue
        clean_sentences.append(s.strip())

    text_vector = sentence_to_vec(text, 100)[0]

    sentences_vector = sentence_to_vec(clean_sentences, 100)

    vectors_sentences_dict = dict(zip(sentences_vector, clean_sentences))

    sorted_vectors = sorted(sentences_vector, reverse=True, key=lambda vec: cosine_distance(text_vector, vec))[:3]
    abstract_sentences = []
    for v in sorted_vectors:
        abstract_sentences.append(vectors_sentences_dict[v])

    return abstract_sentences

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/api/abstract", methods=['POST'])
def abstract():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    return json.dumps(parse_text(data['text']))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=False)