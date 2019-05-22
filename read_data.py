# -*- coding: utf-8 -*-

import os
import logging
import spacy
import nltk
import pandas as pd
import pickle as pkl

from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

from parse_icd10 import ICD10Hierarchy
from ext.CharSplit import char_split
from google.cloud import translate


class DataReader:
    
    def __init__(self):
        self.data_dir = "data"
    
    def read_doc(self, fname):
        with open(fname, mode="r", encoding="utf-8", errors="ignore") as rf:
            data = list()
            for line in rf:
                line = line.strip()
                if line:
                    data.append(line)
            return data 
    
    def read_docs(self, train_or_test):
        if train_or_test == "train":
            docs_dir = os.path.join(self.data_dir, "nts-icd", "docs-training")
        else:
            # test set not released yet
            docs_dir = os.path.join(self.data_dir, "nts-icd", "test", "docs")
        
        incompl_count = 0
        for datafile in os.listdir(docs_dir):
            if datafile == "id.txt":
                continue
            
            # filename is uid
            doc_id = int(datafile.rstrip(".txt"))
            data = self.read_doc(os.path.join(docs_dir, datafile))
            
            # sanity check: each file must have 6 lines of text as in README
            if len(data) != 6:
                incompl_count += 1
            
            # use special token to recover each text field (if needed)
            data = "<SECTION>".join(data)
            
            yield doc_id, data
        
        # report if incompletes
        if incompl_count > 0:
            print("[INFO] %d docs do not have 6 text lines" % incompl_count)
    
    def read_anns(self):
        anns_file = os.path.join(self.data_dir, "nts-icd", "anns_train_dev.txt")
        
        with open(anns_file) as rf:
            for line in rf:
                line = line.strip()
                if line:
                    doc_id, icd10_codes = line.split("\t")
                    # sanity check: remove any duplicates, if there
                    yield int(doc_id), set(icd10_codes.split("|"))
    
    def read_ids(self, ids_file):
        ids_file = os.path.join(self.data_dir, "nts-icd", ids_file)
        ids = set()
        
        with open(ids_file, "r") as rf:
            for line in rf:
                line = line.strip()
                if line:
                    if line == "id": # line 242 in train ids
                        continue
                    ids.add(int(line))
        
        return ids
    
    def read_data(self, train_test):
        def read(docs, anns, ids):
            id2anns = {a[0]:a[1] for a in anns if a[0] in ids}
            # list of tuple (doc text, doc id, set of annotations)
            data = [(d[1], d[0], id2anns[d[0]]) for d in docs if d[0] in id2anns]
            return data
        
        if train_test == "train":
            # load training-dev common data
            docs_train_dev = list(self.read_docs("train"))
            anns_train_dev = list(self.read_anns())
            
            print("[INFO] num of annotations in `anns_train_dev.txt`: %d" % len(anns_train_dev))
            
            # train data
            ids_train = self.read_ids("ids_training.txt")
            data_train = read(docs_train_dev, anns_train_dev, ids_train)
            
            # dev data
            ids_dev = self.read_ids("ids_development.txt")
            data_dev = read(docs_train_dev, anns_train_dev, ids_dev)
                
            return data_train, data_dev
        
        else:
            # load test docs and annotations
            data_test = list(self.read_docs("test"))
            
            return data_test


class TextProcessor:
    
    def __init__(self, en_translate=False, split_compound=False):
        # spacy word tokenizers
        self.word_tokenizers = {
            "de": spacy.load('de_core_news_sm', disable=['tagger', 'parser', 'ner']).tokenizer,
            "en": spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner']).tokenizer
        }
        # nltk sent tokenizers
        self.sent_tokenizers = {
            "de": nltk.data.load('tokenizers/punkt/german.pickle').tokenize,
            "en": nltk.data.load('tokenizers/punkt/english.pickle').tokenize
        }
        # special tokens
        self.sent_sep_tok = "<SENT>"
        self.section_sep_tok = "<SECTION>"
        # google translator
        self.en_translate = en_translate
        if en_translate:
            self.translate_client = translate.Client()
        self.split_compound = split_compound      
    
    def process_doc(self, doc):
        doc = doc.split(self.section_sep_tok) # returns each section
        doc_de = list()
        if self.en_translate:
            doc_en = list()
        
        for textfield in doc:
            sents_de = list(self.sents_tokenize(textfield, "de"))
            sents_tokens_de = list()
            for sent in sents_de:
                tokenized_text = " ".join(list(self.words_tokenize(sent, "de")))
                sents_tokens_de.append(tokenized_text)
            sents_tokens_de = self.sent_sep_tok.join(sents_tokens_de)
            doc_de.append(sents_tokens_de)
            
            if self.en_translate:
                sents_en = list(self.sents_tokenize(self.translate(textfield), "en"))
                sents_tokens_en = list()
                for sent in sents_en:
                    tokenized_text = " ".join(list(self.words_tokenize(sent, "en")))
                    sents_tokens_en.append(tokenized_text)
                sents_tokens_en = self.sent_sep_tok.join(sents_tokens_en)
                doc_en.append(sents_tokens_en)
        
        doc_de = self.section_sep_tok.join(doc_de)
        if self.en_translate:
            doc_en = self.section_sep_tok.join(doc_en)
            return doc, doc_de, doc_en
        return doc, doc_de
    
    def sents_tokenize(self, text, lang):
        for sent in self.sent_tokenizers[lang](text):
            sent = sent.strip()
            if sent:
                yield sent
    
    def words_tokenize(self, text, lang):
        for token in self.word_tokenizers[lang](text):
            token = token.text.strip()
            if token:
                yield token
    
    def translate(self, text):
        return self.translate_client.translate(text, target_language="en")['translatedText']
    
    @staticmethod
    def de_compounds_split(word, t=0.8):
        res = char_split.split_compound(word)[0]
        if res[0] >= t:
            return res[1:]
        else:
            return word
    
    def process_with_context(self, text_and_context):
        text = text_and_context[0]
        context = text_and_context[1:]
        return tuple([self.process_doc(text)] + list(context))
    
    def mp_process(self, data, max_workers=8, chunksize=512):
        """
        data : tup(doc id, doc text, labels list)
        """
        ret = list()
        if max_workers <= 1:
            for idx, item in enumerate(data):
                if idx % 100 == 0 and idx != 0:
                    print("[INFO]: {} documents proceesed".format(idx))
                ret.append(self.process_with_context(item))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                emap = executor.map(self.process_with_context, data, chunksize=chunksize)
                for idx, result in enumerate(emap):
                    if idx % 100 == 0 and idx != 0:
                        print("[INFO]: {} documents proceesed".format(idx))
                    ret.append(result)
        return ret


def save(fname, data):
    with open(fname, "wb") as wf:
        pkl.dump(data, wf)


def prepare_processed():
    # translation takes time and translated texts already in tmp/ dir
    TRANSLATE = False
    # uncomment if translation is required
    if TRANSLATE:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/mlt/saad/tmp/translate-de-to-en-eacaff80b066.json"
    os.makedirs("tmp", exist_ok=True)
    dr = DataReader()
    train_data, dev_data = dr.read_data("train")
    tp = TextProcessor(translate=TRANSLATE)
    # each processed record is:
    # [(doc_orig, doc_de, doc_en), doc_id, {doc_labels}]
    # where: 
    # doc_orig : is "<SECTION>" separated original document in German
    # doc_de : is "<SECTION" separated, where each section is tokenized to sentences
    #          by "<SENT>" separator and each sentence is further tokenized separated
    #          by whitespace. (again German).
    # doc_en : same as doc_de expect the text is translated to English. This field is
    #          absent if translation is turned off
    # doc_id : is unique document id as obtained from task's filenames
    # doc_labels : is a set of original document labels
    train_data = tp.mp_process(train_data)
    dev_data = tp.mp_process(dev_data)
    # save data
    tfname = os.path.join("tmp", "train_data_de" + "_en.pkl" if TRANSLATE else ".pkl")
    dfname = os.path.join("tmp", "dev_data_de" + "_en.pkl" if TRANSLATE else ".pkl")
    save(tfname, train_data)
    save(dfname, dev_data)


def read_train_file(processed_train_file, label_threshold=25):
    with open(processed_train_file, "rb") as rf:
        train_data = pkl.load(rf)
    
    count_labels = Counter([label for val in train_data for label in val[-1]])
    print("[INFO] top most frequent labels:", count_labels.most_common(10))
    
    if label_threshold is None:
        discard_labels = set()
    else:
        discard_labels = {k for k, v in count_labels.items() if v < label_threshold}
    temp = []
    for val in train_data:
        val_labels = {label for label in val[-1] if label not in discard_labels}
        if val_labels:
            val[-1] = val_labels
            temp.append(val)
    if discard_labels:
        print(
            "[INFO] discarded %d labels with counts less than %d; remaining labels %d" 
            % (len(discard_labels), label_threshold, len(count_labels) - len(discard_labels))
        )
        print("[INFO] no. of data points removed %d" % (len(train_data) - len(temp)))
    
    train_data = temp[:]
    mlb = MultiLabelBinarizer()
    temp = [val[-1] for val in train_data]
    labels = mlb.fit_transform(temp)
    train_data = [(val[0], val[1], labels[idx, :]) for idx, val in enumerate(train_data)]
    return train_data, mlb, discard_labels


def read_dev_file(proceesed_dev_file, mlb, discard_labels):
    with open(proceesed_dev_file, "rb") as rf:
        dev_data = pkl.load(rf)
    
    count_labels = Counter([label for val in dev_data for label in val[-1]])
    print("[INFO] top most frequent labels:", count_labels.most_common(10))
    temp = []
    for val in dev_data:
        # discard any labels and keep only ones seen in training
        val_labels = {
            label for label in val[-1] 
            if label not in discard_labels and label in set(mlb.classes_)
        }
        if val_labels:
            val[-1] = val_labels
            temp.append(val)
    print("[INFO] no. of data points removed %d" % (len(dev_data) - len(temp)))
    
    dev_data = temp[:]
    temp = [val[-1] for val in dev_data]
    labels = mlb.transform(temp)
    dev_data = [(val[0], val[1], labels[idx, :]) for idx, val in enumerate(dev_data)]
    return dev_data


def prepare_varying_train_data(train_file, dev_file):
    thresholds = [None, 5, 10, 15, 20, 25, 50]
    os.makedirs("tmp", exist_ok=True)
    for t in thresholds:
        train_data, mlb, discard_labels = read_train_file(train_file, t)
        dev_data = read_dev_file(dev_file, mlb, discard_labels)
        if t is None:
            t = 0
        suffix = "_t{}_c{}.pkl".format(t, len(mlb.classes_))
        save(os.path.join("tmp", "train_data"+suffix), train_data)
        save(os.path.join("tmp", "dev_data"+suffix), dev_data)
        save(os.path.join("tmp", "mlb"+suffix), mlb)
        save(os.path.join("tmp", "discarded"+suffix), discard_labels)
