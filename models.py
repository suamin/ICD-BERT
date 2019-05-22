# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from load_data import load_pkl_datafile


MASK_VALUE = -1e18


class Baseline:
    
    def __init__(self, train_file, dev_file, use_data="de"):
        self.model = OneVsRestClassifier(LinearSVC())
        self.vectorizer = TfidfVectorizer(max_features=10000)
        
        train_data = load_pkl_datafile(train_file, use_data=use_data, as_sents=False)
        train_docs = [d[0] for d in train_data]
        self.Xtrain = self.vectorizer.fit_transform(train_docs)
        self.ytrain = [d[1] for d in train_data]
        self.ytrain = np.array(self.ytrain)
        
        dev_data = load_pkl_datafile(dev_file, use_data=use_data, as_sents=False)
        dev_docs = [d[0] for d in dev_data]
        self.Xdev = self.transform_docs(dev_docs)
        self.ydev = [d[1] for d in dev_data]
        self.ydev = np.array(self.ydev)
        self.dev_ids = [d[2] for d in dev_data]
    
    def train(self):
        self.model.fit(self.Xtrain, self.ytrain)
    
    def transform_docs(self, docs):
        return self.vectorizer.transform(docs)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def eval(self, Xeval, yeval):
        ypred = self.predict(Xeval)
        score = metrics.f1_score(yeval, ypred, average='micro')
        print("Eval score: %0.4f" % score)
        return ypred


class CNN(nn.Module):
    """Convolutional Neural Networks for Sentence Classification [1].
    
    References
    ----------
    [1] Convolutional Neural Networks for Sentence Classification
     *  https://github.com/Shawn1993/cnn-text-classification-pytorch
     *  https://github.com/facebookresearch/pytext/blob/master/pytext/models/representations/docnn.py
    
    """
    
    def __init__(self, V, E, C, Kn=64, Ks=[3, 4, 5], dropout=0.5):
        """
        
        V  | Vocab size
        E  | Embedding dimension
        C  | Number of classes
        Kn | Kernel channels
        Ks | Kernel sizes
        ------------------------
        B  | Batch size
        L  | Max squence length
        
        """
        super().__init__()
        self.embed = nn.Embedding(V, E)
        self.convs = nn.ModuleList([nn.Conv1d(E, Kn, K, padding=K) for K in Ks])
        # output layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(len(Ks) * Kn, C)
    
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))         # B x E x L -> B x Kn x *
        x, _ = torch.max(x, dim=2)  # B x Kn
        return x
    
    def forward(self, x):
        x = self.embed(x)           # B x L -> B x L x E :[out]
        # swap L and E as nn.Conv1d expects a tensor of shape: B x E x L
        x = x.transpose(1, 2)
        x = [self.conv_and_pool(x, conv) for conv in self.convs]
        x = torch.cat(x, 1)         # B x len(Ks)*Kn
        
        # output layer
        x = self.dropout(x)
        logits = self.linear(x)     # B x len(Ks)*Kn -> B x C
        
        return logits


class HAN(nn.Module):
    """Hierarchical Attention Network for document classification [1].
    
    References
    ----------
    [1] https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    
    """
    
    def __init__(self, V, E, C, h=50, L=25, T=50, bidirectional=True, dropout=0.5):
        super().__init__()
        """
        
        V  | Vocab size
        E  | Embedding dimension
        C  | Number of classes
        h  | Hidden dimension
        ------------------------
        B  | Batch size
        L  | Max no. of sentences in a document
        T  | Max no. of words in a sentence
        H  | 2*h or h depending on bidirectional
        
        """
        self.embed = nn.Embedding(V, E)
        self.word_encoder = nn.GRU(
            input_size=E,
            hidden_size=h,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.H = 2*h if bidirectional else h
        self.L = L
        self.T = T
        
        # word encoder + attention
        self.word_attention = nn.Parameter(torch.randn([self.H, 1]).float())
        # u_w in paper
        self.word_linear = nn.Linear(
            in_features=self.H,
            out_features=self.H
        )
        
        # sentence encoder + attention
        self.sentence_encoder = nn.GRU(
            input_size=self.H,
            hidden_size=h,
            bidirectional=bidirectional,
            batch_first=True
        )
        # u_s in paper
        self.sentence_attention = nn.Parameter(torch.randn([self.H, 1]).float())
        self.sentence_linear = nn.Linear(
            in_features=self.H,
            out_features=self.H
        )
        
        # output layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.H, C)
    
    def forward(self, x):
        B = x.shape[0]
        words_mask = x != 0                             # B x L x T
        
        # reshape x to represent words only
        x = x.view(B*self.L, self.T)                    # BL x T
        # reshape the mask as well
        words_mask = words_mask.view(B*self.L, self.T)  # BL x T
        x = self.embed(x)                               # BL x T x E
        
        # ===========================
        # word encoding and attention
        # ===========================
        h_it, _ = self.word_encoder(x)                  # BL x T x H 
        u_it = torch.tanh(self.word_linear(h_it))       # BL x T x H
        # reshape word_attention from H x 1 -> BL x H x 1
        u_w = self.word_attention.unsqueeze(0).expand(B*self.L, -1, -1)
        s_i = u_it.bmm(u_w)                             # (BL x T x H) x (BL x H x 1) -> BL x T x 1
        # apply mask
        words_mask = words_mask.unsqueeze(-1)
        s_i = s_i.masked_fill(~words_mask, MASK_VALUE)
        # attention vector
        a_it = torch.softmax(s_i, dim=1)                # BL x T x 1
        a_it = a_it.transpose(-1, -2)                   # BL x 1 x T
        x_ = a_it.bmm(h_it)                             # (BL x 1 x T) x (BL x T x H) -> BL x 1 x H
        # residual
        x = h_it.mean(dim=1).unsqueeze(dim=1) + x_
        
        # re-adjust x to be of shape: B x L x H
        x = x.squeeze(1).view(B, self.L, self.H)        # B x L x H
        
        # ===============================
        # sentence encoding and attention
        # ===============================
        h_i, _ = self.sentence_encoder(x)               # B x L x H
        u_i = torch.tanh(self.sentence_linear(h_i))     # B x L x H
        # reshape sentence_attention from H x 1 -> B x H x 1
        u_s = self.sentence_attention.unsqueeze(0).expand(B, -1, -1)
        v = u_i.bmm(u_s)                                # (B x L x H) x (B x H x 1) -> B X L X 1
        # get sentence mask
        words_mask = words_mask.view(B, self.L, -1)     # B x L x T
        # sentence padding is where all indicies sum up to 0
        sents_mask = words_mask.sum(-1) != 0            # B x L 
        sents_mask = sents_mask.unsqueeze(-1)           # B x L x 1
        # apply mask
        v = v.masked_fill(~sents_mask, MASK_VALUE)
        a_i = torch.softmax(v, dim=1) 
        x_ = a_i.transpose(-1, -2).bmm(h_i)             # (B x 1 x L) x (B x L x H) -> B x 1 x H
        # residual
        x = h_i.mean(dim=1).unsqueeze(dim=1) + x_
        x = x.squeeze(1)                                # B x H
        
        # output layer
        x = self.dropout(x)
        logits = self.linear(x)                         # B x H -> B x C
        
        return logits


class SelfAttentionLSTM(nn.Module):
    
    def __init__(self, V, E, C, h=50, bidirectional=True, dropout=0.5):
        """
        
        V  | Vocab size
        E  | Embedding dimension
        C  | Number of classes
        h  | Hidden dimension
        ------------------------
        B  | Batch size
        L  | Max squence length
        H  | 2*h or h depending on bidirectional
        
        """
        super().__init__()
        self.embed = nn.Embedding(V, E)
        self.encoder = nn.LSTM(
            input_size=E,
            hidden_size=h,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.H = 2*h if bidirectional else h
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=self.H, out_features=C)
    
    def forward(self, x):
        mask = x != 0                               # B x L
        x = self.embed(x)                           # B x L x E
        x, _ = self.encoder(x)                      # B x L x H
        
        # self-attention
        attn_scores = x.bmm(x.transpose(-1, -2))    # (B x L x H) x (B x H x L) -> B x L x L
        mask = mask.unsqueeze(-1).expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill(~mask, MASK_VALUE)
        attn = torch.softmax(attn_scores, dim=-1)   # B x L x L
        
        x = x + attn.bmm(x)                         # (B x L x L) x (B x L x H) -> B x L x H
        x = x.mean(dim=1)                           # B x H
        
        # output layer
        x = self.dropout(x)
        logits = self.linear(x)                     # B x H -> B x C
        
        return logits


class ICDCodeAttentionLSTM(nn.Module):
    
    def __init__(self, V, E, C, T, Tv, h=50, bidirectional=True, dropout=0.5):
        """
        
        V  | Vocab size
        E  | Embedding dimension
        C  | Number of classes
        h  | Hidden dimension
        ------------------------
        B  | Batch size
        L  | Max squence length
        H  | 2*h or h depending on bidirectional
        T  | Titles matrix of shape Tn x Tl
        Tn | No. of titles
        Tl | Max title squence length
        Tv | Titles vocab size
        
        """
        super().__init__()
        self.embed = nn.Embedding(V, E)
        self.encoder = nn.LSTM(
            input_size=E,
            hidden_size=h,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.embed_T = nn.Embedding(Tv, E)
        self.encoder_T = nn.LSTM(
            input_size=E,
            hidden_size=h,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.H = 2*h if bidirectional else h
        self.T = T
        self.C = C
        
        # output layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=self.H, out_features=C)
    
    def forward(self, x):
        B = x.shape[0]
        mask = x != 0                                   # B x L
        
        x = self.embed(x)                               # B x L x E
        x, _ = self.encoder(x)                          # B x L x H
        
        Te_mask = (self.T != 0).float()                 # Tn x Tl
        Te_mask = Te_mask.mean(dim=1) != 0              # Tl
        Te_mask = Te_mask.unsqueeze(0).expand(B, -1)    # B x Tl
        Te = self.embed_T(self.T)                       # Tn x Tl -> Tn x Tl x E
        Te, _ = self.encoder_T(Te)                      # Tn x Tl x H
        Te = Te.mean(dim=1)                             # Tn x H
        Te = Te.unsqueeze(0).expand(B, -1, -1)          # Tn x H -> 1 x Tn x H -> B x Tn x H
        x = torch.cat([x, Te], 1)                       # B x L' x H (where, L' = L + Tn)
        
        # self-attention
        attn_scores = x.bmm(x.transpose(-1, -2))        # (B x L' x H) x (B x H x L') -> B x L' x L'
        mask = torch.cat([mask, Te_mask], 1).unsqueeze(-1).expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill(~mask, MASK_VALUE)
        attn = torch.softmax(attn_scores, dim=-1)       # B x L x L
        
        x = x + attn.bmm(x)                             # (B x L x L) x (B x L x H) -> B x L x H
        x = x.mean(dim=1)                               # B x H
        
        # output layer
        x = self.dropout(x)
        logits = self.linear(x)                         # B x H -> B x C
        
        return logits


### other models


class HierarchicalSelfAttention(nn.Module):
    """Hierarchical Attention Network for document classification [1].
    
    References
    ----------
    [1] https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    
    """
    
    def __init__(self, V, E, C, h=50, L=10, T=40, bidirectional=True, dropout=0.5):
        super().__init__()
        """
        
        V  | Vocab size
        E  | Embedding dimension
        C  | Number of classes
        h  | Hidden dimension
        ------------------------
        B  | Batch size
        L  | Max no. of sentences in a document
        T  | Max no. of words in a sentence
        H  | 2*h or h depending on bidirectional
        
        """
        self.embed = nn.Embedding(V, E)
        self.word_encoder = nn.GRU(
            input_size=E,
            hidden_size=h,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.H = 2*h if bidirectional else h
        self.L = L
        self.T = T
        
        # sentence encoder + attention
        self.sentence_encoder = nn.GRU(
            input_size=self.H,
            hidden_size=h,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # output layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.H, C)
    
    def forward(self, x):
        B = x.shape[0]
        # B x L x T
        words_mask = x != 0
        # reshape x to represent words only: BL x T
        x = x.view(B*self.L, self.T)
        # reshape the mask as well: BL x T
        words_mask = words_mask.view(B*self.L, self.T)
        
        # [in]: BL x T -> BL x T x E :[out]
        x = self.embed(x)
        
        # ===========================
        # word encoding and attention
        # ===========================
        # [in]: BL x T x E -> BL x T x H :[out]
        x, _ = self.word_encoder(x)
        
        # (BL x T x H) x (BL x H x T) -> BL x T x T
        attn_scores = x.bmm(x.transpose(-1, -2))
        # BL x T -> BL x T x T
        mask = words_mask.unsqueeze(-1).expand_as(attn_scores)
        attn_scores = attn_scores.masked_fill(~mask, MASK_VALUE)
        # BL x T x T
        attn = torch.softmax(attn_scores, dim=-1)
        
        # (BL x T x T) x (BL x T x H) -> BL x T x H
        x = x + attn.bmm(x)
        # BL x H
        x = x.mean(dim=1)
        # B x L x H
        x = x.view(B, self.L, -1)
        
        # ===============================
        # sentence encoding and attention
        # ===============================
        # [in]: B x L x H -> B x L x H :[out]
        x, _ = self.sentence_encoder(x)
        
        # (B x L x H) x (B x H x L) -> B x L x L
        attn_scores = x.bmm(x.transpose(-1, -2))
        # BL x T -> B x L x T -> B x L
        sents_mask = words_mask.view(B, self.L, -1).sum(-1) != 0
        # B x L -> B x L x H
        sents_mask = sents_mask.unsqueeze(-1).expand(B, self.L, -1)
        attn_scores = attn_scores.masked_fill(~sents_mask, MASK_VALUE)
        # B x L x L
        attn = torch.softmax(attn_scores, dim=-1)
        
        # (B x L x L) x (B x L x H) -> B x L x H
        x = x + attn.bmm(x)
        # B x H
        x = x.mean(dim=1)
        
        # output layer
        x = self.dropout(x)
        # [in]: B x H -> B x C :[out]
        logits = self.linear(x)
        
        return logits
