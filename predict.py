# -*- coding: utf-8 -*-

import torch
import logging
import os

import numpy as np
import pickle as pkl


from bert_multilabel_run_classifier import BertForMultiLabelSequenceClassification
from bert_multilabel_run_classifier import BertTokenizer, ClefTask1Processor
from bert_multilabel_run_classifier import convert_examples_to_features, sigmoid

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import SequentialSampler

from tqdm import tqdm, trange
from sklearn import metrics

import models
import matplotlib.pyplot as plt

from load_data import load_pkl_datafile
from load_data import get_data, batched_data, get_titles_T
from train import evaluate

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("predictions")


BASE_DIR = "exps-data"
MODELS_BASE_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def bert_predict(bert_model_dir, test_or_dev, use_data="en", 
                 max_seq_length=256, batch_size=16, return_logits=False,
                 data_dir=DATA_DIR, device=DEVICE):
    """Run BERT based models on test or dev set using original
    or translated texts.
    
    """
    tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=False)
    processor = ClefTask1Processor(data_dir, use_data=use_data)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    model = BertForMultiLabelSequenceClassification.from_pretrained(bert_model_dir, num_labels=num_labels)
    model.to(device)
    
    if test_or_dev == "test":
        examples = processor.get_test_examples()
    else:
        examples = processor.get_dev_examples()
    features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer)
    
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", batch_size)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_doc_ids = torch.tensor([f.guid for f in features], dtype=torch.long)
    
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_doc_ids)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    
    model.eval()
    preds = []
    ids = []
    if test_or_dev == "test":
        ids_file = os.path.join(data_dir, "ids_test.txt")
    else:
        ids_file = os.path.join(data_dir, "ids_development.txt")
    all_ids_test = read_ids(ids_file)
    
    for input_ids, input_mask, segment_ids, doc_ids in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        doc_ids = doc_ids.to(device)
        
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        
        if torch.cuda.is_available():
            logits = logits.detach().cpu().numpy()
            doc_ids = doc_ids.detach().cpu().numpy()
        
        if len(preds) == 0:
            preds.append(logits)
        else:
            preds[0] = np.append(preds[0], logits, axis=0)
        
        if len(ids) == 0:
            ids.append(doc_ids)
        else:
            ids[0] = np.append(ids[0], doc_ids, axis=0)
    
    ids = ids[0]
    preds = preds[0]
    if not return_logits:
        preds = sigmoid(preds)
        preds = (preds > 0.5).astype(int)
    
    preds = [preds[i, :] for i in range(preds.shape[0])]
    id2preds = {val:preds[i] for i, val in enumerate(ids)}
    
    # for i, val in enumerate(all_ids_test):
    #     if val not in id2preds:
    #         id2preds[val] = []
    
    return id2preds


def get_test_data(model_name, lang, max_seq_len=256, batch_size=64, data_dir=DATA_DIR):
    if model_name == "han":
        as_heirarchy = True
    else:
        as_heirarchy = False
    titles_vocab_size = 0
    if model_name == "clstm":
        if lang == "en":
            codes_titles_file = os.path.join(BASE_DIR, "codes_and_titles_en.txt")
        else:
            codes_titles_file = os.path.join(BASE_DIR, "codes_and_titles_de.txt")
        T, titles_word2index = get_titles_T(codes_titles_file)
        titles_vocab_size = len(titles_word2index)
    else:
        T = None
    
    train_file = os.path.join(data_dir, "train_data.pkl")
    dev_file = os.path.join(data_dir, "dev_data.pkl")
    test_file = os.path.join(data_dir, "test_data.pkl")
    
    _, dev_data, test_data, word2index = get_data(
        train_file, dev_file, use_data=lang, max_seq_len=max_seq_len,
        as_heirarchy=as_heirarchy, max_sents_in_doc=10, 
        max_words_in_sent=40, test_file=test_file
    )
    
    # dev data
    Xdev, ydev, ids_dev = dev_data
    vocab_size = len(word2index)
    num_classes = ydev[0].shape[0]
    # test data
    Xtest, ids_test = test_data
    
    dev_dataloader = batched_data(Xdev, ids_dev, batch_size=batch_size)
    test_dataloader = batched_data(Xtest, ids_test, batch_size=batch_size)
    
    return test_dataloader, dev_dataloader, vocab_size, titles_vocab_size, num_classes, T


def generate_preds(preds_file, id2preds):
    with open(preds_file, "w") as wf:
        for doc_id, preds in id2preds.items():
            line = str(doc_id) + "\t" + "|".join(preds) + "\n"
            wf.write(line)


def challenge_eval(dev_or_test, preds_file, out_file, data_dir=DATA_DIR):
    eval_cmd = 'python evaluation.py --ids_file="{}" --anns_file="{}" --dev_file="{}" --out_file="{}"'
    if dev_or_test == "dev":
        ids_file = os.path.join(data_dir, "ids_development.txt")
        anns_file = os.path.join(data_dir, "anns_train_dev.txt")
    else:
        ids_file = os.path.join(data_dir, "ids_test.txt")
        anns_file = os.path.join(data_dir, "anns_test.txt")
    eval_cmd = eval_cmd.format(ids_file, anns_file, preds_file, out_file)
    eval_results = os.popen(eval_cmd).read()
    return eval_results


def get_gold_dev_test(data_dir=DATA_DIR):
    ids_test = read_ids (os.path.join(data_dir, "ids_test.txt"))
    with open(os.path.join(data_dir, "mlb.pkl"), "rb") as rf:
        mlb = pkl.load(rf)
    gold_test = {}
    with open(os.path.join(data_dir, "anns_test.txt")) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            test_id, test_labels = line.split("\t")
            test_id = int(test_id)
            test_labels = test_labels.split("|")
            test_labels_y = mlb.transform([set(test_labels)])
            gold_test[test_id] = test_labels_y.tolist()[0]
    
    ids_dev = read_ids (os.path.join(data_dir, "ids_development.txt"))
    with open(os.path.join(data_dir, "mlb.pkl"), "rb") as rf:
        mlb = pkl.load(rf)
    gold_dev = {}
    with open(os.path.join(data_dir, "anns_train_dev.txt")) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            test_id, test_labels = line.split("\t")
            test_id = int(test_id)
            # skip train ids
            if test_id not in ids_dev:
                continue
            test_labels = test_labels.split("|")
            test_labels_y = mlb.transform([set(test_labels)])
            gold_dev[test_id] = test_labels_y.tolist()[0]
    return gold_test, ids_test, gold_dev, ids_dev


def others_predict(gold_test, ids_test, max_seq_length=256, 
                   batch_size=64, data_dir=DATA_DIR, models_dir=MODELS_BASE_DIR, 
                   results_dir=RESULTS_DIR, device=DEVICE):
    model_names = ["cnn", "han", "slstm", "clstm"]
    langs = ["en", "de"]
    embs = {"en": ["fasttext", "pubmed"], "de": ["fasttext"]}
    
    # load multi-label binarizer that contains classes and their labels mapping
    with open(os.path.join(data_dir, "mlb.pkl"), "rb") as rf:
        mlb = pkl.load(rf)
    
    for lang in langs:
        for emb in embs[lang]:
            if emb == "fasttext":
                embed_dim = 300
            else:
                embed_dim = 400
            hidden_dim = 300
            for model_name in model_names:
                test_loader, dev_loader, V, Tv, C, T = get_test_data(
                    model_name, lang, max_seq_len=256, 
                    batch_size=batch_size, data_dir=data_dir
                )
                vocab_size = V
                titles_vocab_size = Tv
                num_classes = C
                if model_name == "cnn":
                    model = models.CNN(
                        vocab_size, embed_dim, num_classes
                    )
                elif model_name == "han":
                    model = models.HAN(
                        vocab_size, embed_dim, num_classes, 
                        h=hidden_dim, L=10, T=40, bidirectional=True
                    )
                elif model_name == "slstm":
                    model = models.SelfAttentionLSTM(
                        vocab_size, embed_dim, num_classes, 
                        h=hidden_dim, bidirectional=True
                    )
                else:
                    if emb == "fasttext" and lang == "de":
                        continue
                    if lang == "de":
                        hidden_dim = 150
                    T = T.to(device)
                    model = models.ICDCodeAttentionLSTM(
                        vocab_size, embed_dim, num_classes, T,
                        Tv=titles_vocab_size, h=hidden_dim, 
                        bidirectional=True
                    )
                model.to(device)
                model_name = "-".join([model_name, emb, lang])
                print("______________________________________")
                print("          {}                  ".format(model_name))
                print("______________________________________")
                model_dir = os.path.join(models_dir, model_name)
                model_file = os.path.join(model_dir, "model.pt")
                model.load_state_dict(torch.load(model_file))
                model.eval()
                
                _, (_, test_preds, _, test_ids, _) = evaluate(test_loader, model, device, no_labels=True)
                testid2preds = {
                    i: mlb.classes_[test_preds[idx].astype(bool)].tolist()
                    for idx, i in enumerate(test_ids)
                }
                
                # official (include preds for doc ids where we do not even have gold labels)
                # this badly affects model as model make predictions for those examples as
                # well giving all as false positives, hurting precision badly.
                test_preds_official = {
                    k:testid2preds[k] if k in testid2preds else [] 
                    for k in ids_test
                }
                preds_file = os.path.join(results_dir, model_name + "_preds_test.txt")
                generate_preds(preds_file, test_preds_official)
                out_file = os.path.join(results_dir, model_name + "_preds_test_eval.txt")
                results = challenge_eval("test", preds_file, out_file, data_dir)
                print("***** Test results (Original) *****")
                print(results)
                
                # here we only consider evaluating against examples where we have gold labels
                test_preds_fixed = {
                    k:testid2preds[k] if k in testid2preds else [] 
                    for k in set(testid2preds.keys()).intersection(set(gold_test.keys()))
                }
                preds_file = os.path.join(results_dir, model_name + "_preds_test_fixed.txt")
                generate_preds(preds_file, test_preds_fixed)
                out_file = os.path.join(results_dir, model_name + "_preds_test_fixed_eval.txt")
                results = challenge_eval("test", preds_file, out_file, data_dir)
                print("***** Test results (Modified) *****")
                print(results)
                


class Ensemble:
    
    def __init__(self, data_dir=DATA_DIR, models_dir=MODELS_BASE_DIR, 
                 device=DEVICE, plot=True, verbose=True):
        id2scores_m1, id2scores_m2, num_classes = self.get_scores("dev", data_dir, models_dir, device)
        self.k, self.f1 = self.search_k(
            id2scores_m1, id2scores_m2, num_classes, data_dir, plot, verbose
        )
    
    def get_scores(self, dev_or_test, data_dir=DATA_DIR, models_dir=MODELS_BASE_DIR, device=DEVICE):
        if dev_or_test not in ("dev", "test"):
            raise ValueError
        # BioBERT_en 
        id2scores_m1 = bert_predict(
            os.path.join(models_dir, "biobert-en"), 
            test_or_dev=dev_or_test, use_data="en", 
            max_seq_length=256, batch_size=16,
            data_dir=data_dir, device=device,
            return_logits=True
        )
        test_loader, dev_loader, V, Tv, C, T = get_test_data(
            "clstm", "en", max_seq_len=256, 
            batch_size=64, data_dir=data_dir
        )
        vocab_size = V
        titles_vocab_size = Tv
        num_classes = C
        T = T.to(device)
        model = models.ICDCodeAttentionLSTM(
            vocab_size, 400, num_classes, T,
            Tv=titles_vocab_size, h=300, 
            bidirectional=True
        )
        model.load_state_dict(torch.load(os.path.join(MODELS_BASE_DIR, "clstm-pubmed-en", "model.pt")))
        model.to(device)
        model.eval()
        if dev_or_test == "dev":
            data_loader = dev_loader
        else:
            data_loader = test_loader
        _, (scores_m2, _, _, scores_m2_ids, _) = evaluate(data_loader, model, device, no_labels=True)
        id2scores_m2 = {val:scores_m2[i] for i, val in enumerate(scores_m2_ids)}
        return id2scores_m1, id2scores_m2, num_classes
    
    def search_k(self, id2scores_m1, id2scores_m2, num_classes, 
                 data_dir=DATA_DIR, plot=False, verbose=False):
        _, _, gold_dev, ids_dev = get_gold_dev_test(data_dir)
        id2gold = {}
        for i in ids_dev:
            if i in gold_dev:
                id2gold[i] = np.array(gold_dev[i])
            else:
                id2gold[i] = [0.] * num_classes
        
        bests = []
        mj_fs = []
        mj_ps = []
        mj_rs = []
        all_fs = []
        
        for k in np.linspace(0., 1., 50):
            joint_preds = {}
            preds1 = {}
            preds2 = {}
            for i in id2gold:
                # get logits of model 1
                if i in id2scores_m1:
                    logits1 = id2scores_m1[i]
                else:
                    logits1 = np.array([0.] * num_classes)
                # get logits of model 2
                if i in id2scores_m2:
                    logits2 = id2scores_m2[i]
                else:
                    logits2 = np.array([0.] * num_classes)
                
                # weighted logits
                logits = (k * logits1) + ((1-k) * logits2)
                preds = (sigmoid(logits) > 0.5).astype(int)
                joint_preds[i] = preds
                # also collect individual predictions
                preds1[i] = (sigmoid(logits1) > 0.5).astype(int)
                preds2[i] = (sigmoid(logits2) > 0.5).astype(int)
            
            ytrue = np.array([id2gold[i] for i in id2gold])
            ypred = np.array([joint_preds[i] for i in id2gold])
            ypred1 = np.array([preds1[i] for i in id2gold])
            ypred2 = np.array([preds2[i] for i in id2gold])
            
            # ensemble performance metrics
            fj = metrics.f1_score(ytrue, ypred, average='micro')
            pj = metrics.precision_score(ytrue, ypred, average='micro')
            rj = metrics.recall_score(ytrue, ypred, average='micro')
            # model 1 performance metrics
            f1 = metrics.f1_score(ytrue, ypred1, average='micro')
            p1 = metrics.precision_score(ytrue, ypred1, average='micro')
            r1 = metrics.recall_score(ytrue, ypred1, average='micro')
            # model 2 performance metrics
            f2 = metrics.f1_score(ytrue, ypred2, average='micro')
            p2 = metrics.precision_score(ytrue, ypred2, average='micro')
            r2 = metrics.recall_score(ytrue, ypred2, average='micro')
            
            if fj > f1 and fj > f2:
                if verbose:
                    print("\nfound a value with fj > f1 and f2")
                    print("> kappa = %0.3f" % k)
                    print("> ensemble :")
                    print("  P=%0.3f R=%0.3f F1=%0.3f" % (pj, rj, fj))
                    print("> model 1 :")
                    print("  P=%0.3f R=%0.3f F1=%0.3f" % (p1, r1, f1))
                    print("> model 2 :")
                    print("  P=%0.3f R=%0.3f F1=%0.3f" % (p2, r2, f2))
                bests.append((fj, rj, pj, f2, r2, p2, k))
            
            mj_fs.append(fj)
            mj_ps.append(pj)
            mj_rs.append(rj)
            all_fs.extend([fj, f1, f2])
        
        bests = sorted(bests, key=lambda x: x[0], reverse=True)
        best_f1, best_k = bests[0][0], bests[0][-1]
        print("***** best f1-score=%0.3f @ k=%0.3f *****" % (best_f1, best_k))
        
        if plot:
            x = np.linspace(0., 1., 50)
            plt.rc('xtick',labelsize=18)
            plt.rc('ytick',labelsize=18)
            plt.figure(figsize=(15, 10))
            plt.ylim(0.73, 0.95)
            plt.plot(x, mj_fs, label="F1-mirco")
            plt.plot(x, mj_rs, label="Recall", linestyle='--')
            plt.plot(x, mj_ps, label="Precision", linestyle='--')
            plt.plot([bests[0][-1], ] * 2, [0, bests[0][0]], label=r"Best $\kappa$")
            plt.annotate(
                "%0.3f" % bests[0][-1], (bests[0][-1]+ 0.01, 0.73 + 0.005)
            )
            plt.title(
                r"Ensemble parameter $\kappa$ against performance metrics",
                size=25
            )
            plt.xlabel(r"Ensemble parameter $\kappa$ (0-1)", size=20)
            plt.ylabel("Scores", size=20)
            plt.legend(loc="best", fontsize=18)
            plt.show()
        
        return best_k, best_f1
    
    def ensemble_predict(self, data_dir=DATA_DIR, models_dir=MODELS_BASE_DIR, device=DEVICE):
        """Weighted predictions for ensemble model."""
        id2scores_m1, id2scores_m2, _ = self.get_scores("test", data_dir, models_dir, device)
        joint_ids = set(list(id2scores_m1.keys()) + list(id2scores_m2.keys()))
        joint_preds = {}
        # k = 0.6326530612244897
        for i in joint_ids:
            if i in id2scores_m1:
                logits1 = id2scores_m1[i]
            else:
                logits1 = np.array([0.] * num_classes)
            if i in id2scores_m2:
                logits2 = id2scores_m2[i]
            else:
                logits2 = np.array([0.] * num_classes)
            logits = (self.k * logits1) + ( (1-self.k) * logits2)
            preds = (sigmoid(logits) > 0.5).astype(int)
            joint_preds[i] = preds
        return joint_preds


def berts_predict(gold_test, ids_test, data_dir=DATA_DIR, 
                  models_dir=MODELS_BASE_DIR, results_dir=RESULTS_DIR, 
                  device=DEVICE):
    with open(os.path.join(data_dir, "mlb.pkl"), "rb") as rf:
        mlb = pkl.load(rf)
    
    for model_name in ("biobert-en", "bert-en", "multi-bert-de"):
        print("______________________________________")
        print("          {}                  ".format(model_name))
        print("______________________________________")
        lang = model_name.split("-")[-1]
        id2preds = bert_predict(
            os.path.join(models_dir, model_name), 
            test_or_dev="test", use_data=lang, 
            max_seq_length=256, batch_size=16,
            data_dir=data_dir, device=device,
            return_logits=False
        )
        id2preds = {k: mlb.classes_[v.astype(bool)].tolist() for k, v in id2preds.items()}
        # "Original" evaluation
        test_preds_official = {k:id2preds[k] if k in id2preds else [] for k in ids_test}
        preds_file = os.path.join(results_dir, model_name + "_preds_test.txt")
        generate_preds(preds_file, test_preds_official)
        out_file = os.path.join(results_dir, model_name + "_preds_test_eval.txt")
        results = challenge_eval("test", preds_file, out_file, data_dir)
        print("***** Test results (Original) *****")
        print(results)
        # "Modified" evaluation
        test_preds_fixed = {
            k:id2preds[k] if k in id2preds else [] 
            for k in set(id2preds.keys()).intersection(set(gold_test.keys()))
        }
        preds_file = os.path.join(results_dir, model_name + "_preds_test_fixed.txt")
        generate_preds(preds_file, test_preds_fixed)
        out_file = os.path.join(results_dir, model_name + "_preds_test_fixed_eval.txt")
        results = challenge_eval("test", preds_file, out_file, data_dir)
        print("***** Test results (Modified) *****")
        print(results)


def baselines_predict(gold_test, ids_test, data_dir=DATA_DIR, results_dir=RESULTS_DIR):
    with open(os.path.join(data_dir, "mlb.pkl"), "rb") as rf:
        mlb = pkl.load(rf)
    
    for lang in ("en", "de"):
        model_name = "baseline-" + lang
        print("______________________________________")
        print("          {}                  ".format(model_name))
        print("______________________________________")
        bsm = models.Baseline(
            os.path.join(data_dir, "train_data.pkl"),
            os.path.join(data_dir, "dev_data.pkl"),
            use_data=lang
        )
        bsm.train()
        test_data = load_pkl_datafile(
            os.path.join(data_dir, "test_data.pkl"), 
            use_data=lang, 
            as_sents=False
        )
        test_docs = [d[0] for d in test_data]
        Xtest = bsm.vectorizer.transform(test_docs)
        tmp1 = []
        tmp2 = []
        for idx in range(Xtest.shape[0]):
            tmp1.append(idx)
            tmp2.append(test_data[idx][-1])
        Xtest = Xtest[tmp1]
        ypred = bsm.predict(Xtest)
        id2preds = {i:j for i, j in zip(tmp2, ypred)}
        id2preds = {k: mlb.classes_[v.astype(bool)].tolist() for k, v in id2preds.items()}
        # "Original" evaluation
        test_preds_official = {k:id2preds[k] if k in id2preds else [] for k in ids_test}
        preds_file = os.path.join(results_dir, model_name + "_preds_test.txt")
        generate_preds(preds_file, test_preds_official)
        out_file = os.path.join(results_dir, model_name + "_preds_test_eval.txt")
        results = challenge_eval("test", preds_file, out_file, data_dir)
        print("***** Test results (Original) *****")
        print(results)
        # "Modified" evaluation
        test_preds_fixed = {
            k:id2preds[k] if k in id2preds else [] 
            for k in set(id2preds.keys()).intersection(set(gold_test.keys()))
        }
        preds_file = os.path.join(results_dir, model_name + "_preds_test_fixed.txt")
        generate_preds(preds_file, test_preds_fixed)
        out_file = os.path.join(results_dir, model_name + "_preds_test_fixed_eval.txt")
        results = challenge_eval("test", preds_file, out_file, data_dir)
        print("***** Test results (Modified) *****")
        print(results)


def ensembles_predict(gold_test, ids_test, data_dir=DATA_DIR, 
                      models_dir=MODELS_BASE_DIR, results_dir=RESULTS_DIR, 
                      device=DEVICE):
    model = Ensemble(plot=False, verbose=False)
    id2preds = model.ensemble_predict(data_dir=data_dir, models_dir=models_dir, device=device)
    with open(os.path.join(data_dir, "mlb.pkl"), "rb") as rf:
        mlb = pkl.load(rf)
    id2preds = {k: mlb.classes_[v.astype(bool)].tolist() for k, v in id2preds.items()}
    model_name = "ensemble"
    # "Original" evaluation
    test_preds_official = {k:id2preds[k] if k in id2preds else [] for k in ids_test}
    preds_file = os.path.join(results_dir, model_name + "_preds_test.txt")
    generate_preds(preds_file, test_preds_official)
    out_file = os.path.join(results_dir, model_name + "_preds_test_eval.txt")
    results = challenge_eval("test", preds_file, out_file, data_dir)
    print("***** Test results (Original) *****")
    print(results)
    # "Modified" evaluation
    test_preds_fixed = {
        k:id2preds[k] if k in id2preds else [] 
        for k in set(id2preds.keys()).intersection(set(gold_test.keys()))
    }
    preds_file = os.path.join(results_dir, model_name + "_preds_test_fixed.txt")
    generate_preds(preds_file, test_preds_fixed)
    out_file = os.path.join(results_dir, model_name + "_preds_test_fixed_eval.txt")
    results = challenge_eval("test", preds_file, out_file, data_dir)
    print("***** Test results (Modified) *****")
    print(results)


if __name__=="__main__":
    gold_test, ids_test, gold_dev, ids_dev = get_gold_dev_test()
    baselines_predict(gold_test, ids_test)
    berts_predict(gold_test, ids_test)
    others_predict(gold_test, ids_test)
    ensembles_predict(gold_test, ids_test)
