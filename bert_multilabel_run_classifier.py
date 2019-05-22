# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys

import numpy as np
import pickle as pkl

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
from sklearn.metrics import f1_score

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.modeling import BertConfig, BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.modeling import CONFIG_NAME, WEIGHTS_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from torch.nn import BCEWithLogitsLoss
from loss import BalancedBCEWithLogitsLoss


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multi-label classification."""
    
    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) list of string. The labels of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.guid = guid


def load_pkl(fname):
    with open(fname, "rb") as rf:
        return pkl.load(rf)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
    
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return load_pkl(input_file)


class ClefTask1Processor(DataProcessor):
    """Processor for the CLEF eHealth task1 data set."""
    
    def __init__(self, data_dir, use_data="orig", test_file=None):
        self.train_file = os.path.join(data_dir, "train_data.pkl")
        self.dev_file = os.path.join(data_dir, "dev_data.pkl")
        self.test_file = os.path.join(data_dir, "test_data.pkl")
        self.mlb = load_pkl(os.path.join(data_dir, "mlb.pkl"))
        self.use_data = use_data
        #
        data = self._read_tsv(self.train_file)
        y = np.array([i[-1] for i in data])
        self.pos_weight = ((1-y).sum(0) / y.sum(0)).astype(int)
    
    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv(self.train_file))
    
    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv(self.dev_file))
    
    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self._read_tsv(self.test_file))
    
    def get_labels(self):
        return self.mlb.classes_.tolist()
    
    def _create_examples(self, data):
        examples = []
        # each d is tuple of ((doc orig, doc de, doc en [opt]), doc id, binary labels)
        for d in data:
            guid = d[1]
            if self.use_data == "orig":
                text_a = "\n".join(d[0][0].split("<SECTION>"))
            elif self.use_data == "de":
                text_a = "\n".join(d[0][1].replace("<SENT>", "").split("<SECTION>"))
            else:
                text_a = "\n".join(d[0][2].replace("<SENT>", "").split("<SECTION>"))
            text_b = None
            if isinstance(d[-1], np.ndarray):
                labels = d[-1].tolist()
            else:
                labels = []
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,labels=labels))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    
    label_map = {label : i for i, label in enumerate(label_list)}
    
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        tokens_a = tokenizer.tokenize(example.text_a)
        
        tokens_b = None
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_ids = example.labels[:]
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %r" % label_ids)
        
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              guid=example.guid))
    return features


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, num_labels]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    """
    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = BalancedBCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "clef":
        return {"f1": f1_score(y_true=labels, y_pred=preds, average='micro')}
    else:
        raise KeyError(task_name)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


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


def generate_preds_file(dev_dataloader, model, mlb_file, devids_file, preds_file):
    
    with open(mlb_file, "rb") as rf:
        mlb = pkl.load(rf)
    
    all_ids_dev = list(read_ids(devids_file))
    score, preds_data = evaluate(dev_dataloader, model)
    logits, preds, labels, ids, avg_loss = preds_data
    preds = [mlb.classes_[preds[i, :].astype(bool)].tolist() for i in range(preds.shape[0])]
    id2preds = {val:preds[i] for i, val in enumerate(ids)}
    preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(all_ids_dev)]
    
    with open(preds_file, "w") as wf:
        for idx, doc_id in enumerate(all_ids_dev):
            line = str(doc_id) + "\t" + "|".join(preds[idx]) + "\n"
            wf.write(line)
    
    return preds


def main():
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain.pkl files, named: train_data.pkl, "
                        "dev_data.pkl, test_data.pkl and mlb.pkl (e.g. as in `exps-data/data`).")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    ## Other parameters
    parser.add_argument("--use_data",
                        default="orig",
                        type=str,
                        help="Original DE, tokenized DE or tokenized EN.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    processors = {
        "clef": ClefTask1Processor
    }
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    task_name = args.task_name.lower()
    
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    
    processor = processors[task_name](args.data_dir, use_data=args.use_data)
    pos_weight = torch.tensor(processor.pos_weight, requires_grad=False, dtype=torch.float, device="cuda")
    label_list = processor.get_labels()
    num_labels = len(label_list)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples()
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps
        ) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = BertForMultiLabelSequenceClassification.from_pretrained(args.bert_model, 
        cache_dir=cache_dir,
        num_labels=num_labels
    )
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.do_train:
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
        
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    def eval():
        eval_examples = processor.get_dev_examples()
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_doc_ids = torch.tensor([f.guid for f in eval_features], dtype=torch.long)
        
        # output_mode == "classification":
        all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)
        all_label_ids = all_label_ids.view(-1, num_labels)
        
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_doc_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        ids = []
        # FIXME: make it flexible to accept path
        all_ids_dev = read_ids("data/nts-icd/ids_development.txt")
        
        for input_ids, input_mask, segment_ids, label_ids, doc_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            doc_ids = doc_ids.to(device)
            
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            
            # create eval loss and other metric required by the task
            # output_mode == "classification":
            loss_fct = BalancedBCEWithLogitsLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1, num_labels))
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
            if len(ids) == 0:
                ids.append(doc_ids.detach().cpu().numpy())
            else:
                ids[0] = np.append(
                    ids[0], doc_ids.detach().cpu().numpy(), axis=0)
        
        eval_loss = eval_loss / nb_eval_steps
        ids = ids[0]
        preds = sigmoid(preds[0])
        preds = (preds > 0.5).astype(int)
        
        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        #result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss/nb_tr_steps if args.do_train else None
        
        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss
        
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        
        with open(os.path.join(args.data_dir, "mlb.pkl"), "rb") as rf:
            mlb = pkl.load(rf)
        preds = [mlb.classes_[preds[i, :].astype(bool)].tolist() for i in range(preds.shape[0])]
        id2preds = {val:preds[i] for i, val in enumerate(ids)}
        preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(all_ids_dev)]
        
        with open(os.path.join(args.output_dir, "preds_development.txt"), "w") as wf:
            for idx, doc_id in enumerate(all_ids_dev):
                line = str(doc_id) + "\t" + "|".join(preds[idx]) + "\n"
                wf.write(line)

    def predict():
        test_examples = processor.get_test_examples()
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_doc_ids = torch.tensor([f.guid for f in test_features], dtype=torch.long)
        
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_doc_ids)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        ids = []
        # FIXME: make it flexible to accept path
        all_ids_test = read_ids(os.path.join(args.data_dir, "ids_testing.txt"))
        
        for input_ids, input_mask, segment_ids, doc_ids in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            doc_ids = doc_ids.to(device)
            
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
            if len(ids) == 0:
                ids.append(doc_ids.detach().cpu().numpy())
            else:
                ids[0] = np.append(
                    ids[0], doc_ids.detach().cpu().numpy(), axis=0)
        
        ids = ids[0]
        preds = sigmoid(preds[0])
        preds = (preds > 0.5).astype(int)
        
        with open(os.path.join(args.data_dir, "mlb.pkl"), "rb") as rf:
            mlb = pkl.load(rf)
        preds = [mlb.classes_[preds[i, :].astype(bool)].tolist() for i in range(preds.shape[0])]
        id2preds = {val:preds[i] for i, val in enumerate(ids)}
        preds = [id2preds[val] if val in id2preds else [] for i, val in enumerate(all_ids_test)]
        
        with open(os.path.join(args.output_dir, "preds_test.txt"), "w") as wf:
            for idx, doc_id in enumerate(all_ids_test):
                line = str(doc_id) + "\t" + "|".join(preds[idx]) + "\n"
                wf.write(line)
    
    if args.do_train:
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        
        # output_mode == "classification":
        all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
        all_label_ids = all_label_ids.view(-1, num_labels)
        
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                
                # if output_mode == "classification":
                loss_fct = BalancedBCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1, num_labels))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear.get_lr(
                            global_step/num_train_optimization_steps,
                            args.warmup_proportion
                        )
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            eval()
    
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)
        
        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForMultiLabelSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    else:
        model = BertForMultiLabelSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels)
    model.to(device)
    
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval()
        predict()


if __name__ == "__main__":
    main()

"""

export BERT_MODEL=/path/to/bert-model

## STEP 1: Convert TF checkpoint to PyTorch model

python convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path $BERT_MODEL/biobert_model.ckpt \
    --bert_config_file $BERT_MODEL/bert_config.json \
    --pytorch_dump_path $BERT_MODEL/pytorch_model.bin


## STEP 2: Fine-tune the model

export DATA_DIR=exps-data/data
export BERT_EXPS_DIR=tmp/bert-exps-dir

python bert_multilabel_run_classifier.py \
    --data_dir $DATA_DIR \
    --use_data en \
    --bert_model $BERT_MODEL \
    --task_name clef \
    --output_dir $BERT_EXPS_DIR/output \
    --cache_dir $BERT_EXPS_DIR/cache \
    --max_seq_length 256 \
    --num_train_epochs 20 \
    --do_train \
    --do_eval \
    --train_batch_size 6


## STEP 3: Inference

python bert_multilabel_run_classifier.py \
    --data_dir $DATA_DIR \
    --use_data en \
    --bert_model $BERT_EXPS_DIR/output \
    --task_name clef \
    --output_dir $BERT_EXPS_DIR/output \
    --cache_dir $BERT_EXPS_DIR/cache \
    --max_seq_length 256 \
    --do_eval \
    --train_batch_size 6

"""
