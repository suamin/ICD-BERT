# -*- coding: utf-8 -*-

import numpy as np
import torch
import pickle as pkl
import codecs

from collections import Counter
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm, trange
from gensim.models import KeyedVectors


def load_ft_embeds(word2index, embed_dim, pretrain_path):
    #load embeddings
    print('loading word embeddings...')
    mn = 0
    nm = 0
    embedding_matrix = np.zeros((len(word2index), embed_dim))
    f = codecs.open(pretrain_path, encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        if word in word2index:
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix[word2index[word]] = coefs
            mn += 1
        else:
            nm += 1
    print("loaded vectors from `{}`; {} found out of pretrain total {}".format(
        pretrain_path, mn, mn + nm))
    f.close()
    return embedding_matrix


def load_pubmed_gensim_en(word2index):
    print("loading model .bin file ...")
    model = KeyedVectors.load_word2vec_format(
        "/home/mlt/saad/tmp/pubmed2018_w2v_400D/pubmed2018_w2v_400D.bin", binary=True
    )
    print('matching word embeddings...')
    mn = 0
    embedding_matrix = np.zeros((len(word2index), 400))
    for word in word2index:
        if word in model:
            coefs = model[word]
            embedding_matrix[word2index[word]] = coefs
            mn += 1
    print("{} found out of pretrain total".format(mn))
    del model
    return embedding_matrix


def to_fasttext(gmodel, outfile):
    with open(outfile, "w", encoding="utf-8", errors="ignore") as wf:
        wf.write("{} 400\n".format(len(gmodel.wv.vocab), gmodel.vector_size))
        for word in gmodel.wv.vocab:
            wf.write(word + " " + " ".join(["%0.4f" % i for i in gmodel[word]]) + "\n")


def load_pkl_datafile(fname, use_data="de", as_sents=False):
    examples = []
    with open(fname, "rb") as rf:
        data = pkl.load(rf)
        # each item is tuple((doc orig, doc de, doc en [opt]), doc id, binary labels)
        for value, doc_id, one_hot_labels in data:
            if use_data == "orig":
                text = value[0]
                if not as_sents:
                    text = " ".join(text.split("<SECTION>"))
                else:
                    text = text.split("<SECTION>")
            else:
                if use_data == "de":
                    text = value[1]
                else:
                    text = value[2] # en
                if not as_sents:
                    text = " ".join(text.replace("<SENT>", " ").split("<SECTION>"))
                else:
                    text = [s.replace("<SECTION>", "") for s in text.split("<SENT>")]
            examples.append((text, one_hot_labels, doc_id))
    return examples


def build_vocab(texts, min_df=5, max_df=0.6, keep_n=10000):
    counter = Counter([token for text in texts for token in text.split()])
    counter = Counter({k:v for k, v in counter.items() if min_df <= v <= int(len(texts)*max_df)})
    words = [w for w, _ in counter.most_common()[:keep_n-2]]
    word2index = {"<pad>":0, "<unk>": 1}
    for i in range(len(words)):
        word2index[words[i]] = i+2
    return word2index


def text_to_seq(text, word2index):
    return [word2index[token] if token in word2index else word2index["<unk>"] for token in text.split()]


def doc_to_seq(doc, word2index):
    return [text_to_seq(sentence, word2index) for sentence in doc]


def pad_seq(seq, max_len):
    seq = seq[:max_len]
    seq += [0 for i in range(max_len - len(seq))]
    return seq


def pad_doc(doc, max_sents, max_words):
    for idx, sentence in enumerate(doc):
        if len(sentence) != max_words:
            doc[idx] = pad_seq(sentence, max_words)
    if len(doc) < max_sents:
        doc.extend([[0]*max_words for _ in range(max_sents - len(doc))])
    elif len(doc) > max_sents:
        doc = doc[:max_sents]
    return doc


def batched_data(*tensors, batch_size=64):
    data = TensorDataset(*tensors)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def get_X_y_ids(input_file, word2index, use_data="de", max_seq_len=256,
                as_heirarchy=False, max_sents_in_doc=10, max_words_in_sent=40,
                is_test=False):
    
    data = load_pkl_datafile(input_file, use_data=use_data, as_sents=as_heirarchy)
    
    X, y, doc_ids = [], [], []
    for idx, val in enumerate(data):
        text, labels, doc_id = val
        if as_heirarchy:
            X.append(pad_doc(doc_to_seq(text, word2index), max_sents_in_doc, max_words_in_sent))
        else:
            X.append(pad_seq(text_to_seq(text, word2index), max_seq_len))
        
        if not is_test:
            y.append(labels)
        doc_ids.append(doc_id)
    
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float)
    
    doc_ids = torch.tensor(doc_ids, dtype=torch.long)
    
    if as_heirarchy:
        X = X.view(-1, max_sents_in_doc, max_words_in_sent)
    else:
        X = X.view(-1, max_seq_len)
    
    if not is_test:
        num_classes = len(y[0])
        y = y.view(-1, num_classes)
    doc_ids = doc_ids.view(-1)
    
    return X, y, doc_ids


def get_data(train_file, dev_file, use_data="de", max_seq_len=256,
             as_heirarchy=False, max_sents_in_doc=10, max_words_in_sent=40, 
             test_file=None, **kwargs):
    
    data = load_pkl_datafile(train_file, use_data=use_data, as_sents=as_heirarchy)
    if as_heirarchy:
        data = [" ".join(d[0]) for d in data]
    else:
        data = [d[0] for d in data]
    word2index = build_vocab(data, **kwargs)
    
    # train X, y, ids
    Xtrain, ytrain, ids_train = get_X_y_ids(
        train_file, word2index, use_data=use_data, max_seq_len=max_seq_len,
        as_heirarchy=as_heirarchy, max_sents_in_doc=max_sents_in_doc, 
        max_words_in_sent=max_words_in_sent
    )
    # dev X, y, ids
    Xdev, ydev, ids_dev = get_X_y_ids(
        dev_file, word2index, use_data=use_data, max_seq_len=max_seq_len,
        as_heirarchy=as_heirarchy, max_sents_in_doc=max_sents_in_doc, 
        max_words_in_sent=max_words_in_sent
    )
    
    if test_file:
        Xtest, _, ids_test = get_X_y_ids(
            test_file, word2index, use_data=use_data, max_seq_len=max_seq_len,
            as_heirarchy=as_heirarchy, max_sents_in_doc=max_sents_in_doc, 
            max_words_in_sent=max_words_in_sent, is_test=True
        )
    
    if test_file:
        return (Xtrain, ytrain, ids_train), (Xdev, ydev, ids_dev), (Xtest, ids_test), word2index
    else:
        return (Xtrain, ytrain, ids_train), (Xdev, ydev, ids_dev), word2index


def get_titles_T(codes_titles_file):
    titles = []
    with open(codes_titles_file, "r") as rf:
        for line in rf:
            title, code = line.split("\t")
            titles.append(title)
    titles_vocab = build_vocab(titles, 1, 1.0)
    titles = [pad_seq(text_to_seq(i, titles_vocab), 10) for i in titles]
    titles = torch.tensor(titles).long()
    return titles, titles_vocab
