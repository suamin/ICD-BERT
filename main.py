# -*- coding: utf-8 -*-

import os
import pickle as pkl

import torch
import torch.nn as nn

import models
from train import train, evaluate

from load_data import get_data, batched_data, get_titles_T
from load_data import load_ft_embeds, load_pubmed_gensim_en


def read_ids(ids_file):
    ids = set()
    with open(ids_file, "r") as rf:
        for line in rf:
            line = line.strip()
            if line:
                if line == "id": # line 242 in train ids
                    continue
                ids.add(int(line))
    return ids


def generate_preds_file(preds, preds_ids, mlb_file, devids_file, preds_file):
    
    with open(mlb_file, "rb") as rf:
        mlb = pkl.load(rf)
    
    all_ids_dev = list(read_ids(devids_file))
    preds = [mlb.classes_[preds[i, :].astype(bool)].tolist() for i in range(preds.shape[0])]
    id2preds = {val:preds[i] for i, val in enumerate(preds_ids)}
    preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(all_ids_dev)]
    
    with open(preds_file, "w") as wf:
        for idx, doc_id in enumerate(all_ids_dev):
            line = str(doc_id) + "\t" + "|".join(preds[idx]) + "\n"
            wf.write(line)
    
    return preds


def main(train_file, dev_file, lang, model_name, device,
         batch_size=64, max_seq_len=256, embed_dim=300, 
         epochs=50, lr=0.001, load_pretrain_ft=True, 
         load_pretrain_pubmed=False, pretrain_file=None, 
         hidden_dim=300, max_sents_in_doc=10, max_words_in_sent=40, 
         as_heirarchy=False, bidirectional=True):
    
    if lang not in ("de", "en"):
        raise ValueError
    
    if model_name not in ("cnn", "han", "slstm", "clstm"):
        raise ValueError
    
    if load_pretrain_ft:
        load_pretrain_pubmed = False
        embed_dim = 300
    
    if load_pretrain_pubmed:
        embed_dim = 400
    
    if model_name == "han":
        as_heirarchy = True
    else:
        as_heirarchy = False
    
    if model_name == "clstm":
        if lang == "en":
            codes_titles_file = "exps-data/codes_and_titles_en.txt"
        else:
            codes_titles_file = "exps-data/codes_and_titles_de.txt"
        T, titles_word2index = get_titles_T(codes_titles_file)
    
    train_data, dev_data, word2index = get_data(
        train_file, dev_file, use_data=lang, max_seq_len=max_seq_len,
        as_heirarchy=as_heirarchy, max_sents_in_doc=max_sents_in_doc, 
        max_words_in_sent=max_words_in_sent
    )
    
    # training data
    Xtrain, ytrain, ids_train = train_data
    
    vocab_size = len(word2index)
    num_classes = ytrain[0].shape[0]
    
    # dev data
    Xdev, ydev, ids_dev = dev_data
    
    train_dataloader = batched_data(Xtrain, ytrain, ids_train, batch_size=batch_size)
    dev_dataloader = batched_data(Xdev, ydev, ids_dev, batch_size=batch_size)
    
    if load_pretrain_ft and pretrain_file:
        embed_matrix = load_ft_embeds(word2index, embed_dim, pretrain_file)
        if model_name == "clstm":
            embed_matrix_T = load_ft_embeds(titles_word2index, embed_dim, pretrain_file)
    
    elif load_pretrain_pubmed and pretrain_file:
        embed_matrix = load_pubmed_gensim_en(word2index, embed_dim, pretrain_file)
        if model_name == "clstm":
            embed_matrix_T = load_pubmed_gensim_en(titles_word2index, embed_dim, pretrain_file)
    
    else:
        embed_matrix = None
        embed_matrix_T = None
    
    if model_name == "cnn":
        model = models.CNN(
            vocab_size, embed_dim, num_classes
        )
    elif model_name == "han":
        model = models.HAN(
            vocab_size, embed_dim, num_classes, 
            h=hidden_dim, L=max_sents_in_doc, 
            T=max_words_in_sent, bidirectional=bidirectional
        )
    elif model_name == "slstm":
        model = models.SelfAttentionLSTM(
            vocab_size, embed_dim, num_classes, 
            h=hidden_dim, bidirectional=bidirectional
        )
    else:
        model = models.ICDCodeAttentionLSTM(
            vocab_size, embed_dim, num_classes, T,
            Tv=len(titles_vocab), h=hidden_dim, 
            bidirectional=bidirectional
        )
    
    if (load_pretrain_ft or load_pretrain_pubmed) and pretrain_file:
        model.embed.weight.data.copy_(torch.from_numpy(embed_matrix))
        if model_name == "clstm":
            model.embed_T.weight.data.copy_(torch.from_numpy(embed_matrix_T))
    
    model = model.to(device)
    print(model_name)
    print(model)
    
    xavier = False
    if xavier:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    model_save_fname = "./{}_{}.pt".format(lang, model_name)
    
    train(
        train_dataloader, dev_dataloader, model, epochs, lr, 
        device=device, grad_clip=None, model_save_fname=model_save_fname
    )
    
    _, (_, preds, _, ids, _) = evaluate(dev_dataloader, model, device)
    
    return model, model_save_fname, preds, ids


if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, model_save_fname, dev_preds, preds_ids = main(
        train_file="exps-data/data/train_data.pkl",
        dev_file="exps-data/data/dev_data.pkl",
        lang="en",
        load_pretrain_ft=True,
        load_pretrain_pubmed=False,
        pretrain_file="../cc.en.300.vec",
        model_name="slstm",
        device=device
    )
    # pass pretrained model file the path they are
    # > path to "cc.en.300.vec" when `load_ft_embeds` is True
    # > path to "pubmed2018_w2v_400D.bin" when `load_pubmed_gensim_en` is True
    
    torch.save(model.state_dict(), model_save_fname)
    # generate predictions file for evaluation script
    generate_preds_file(
        dev_preds, preds_ids, 
        mlb_file="exps-data/data/mlb.pkl", 
        devids_file="exps-data/data/ids_development.txt", 
        preds_file="./preds_development.txt"
    )
    
    eval_cmd = 'python evaluation.py --ids_file="{}" --anns_file="{}" --dev_file="{}" --out_file="{}"'
    eval_cmd = eval_cmd.format(
        "exps-data/data/ids_development.txt",
        "exps-data/data/anns_train_dev.txt",
        "preds_development.txt",
        "eval_output.txt"
    )
    eval_results = os.popen(eval_cmd).read()
    print("eval results with challenge script:")
    print(eval_results)
