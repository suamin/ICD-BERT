# MLT-DFKI at CLEF eHealth Task 1: Multi-label Classification with BERT #

Code for our submission at [CLEF eHealth Task 1: Multilingual Information Extraction](https://clefehealth.imag.fr/?page_id=171). For details, check [here](http://ceur-ws.org/Vol-2380/paper_67.pdf).

## Requirements
If you're using new [trasnformers](https://github.com/huggingface/transformers) library, then it is recommended to create virtual environment as this code was written with the older version (note there will be no issues even if both versions co-exist):
```bash
pip install pytorch-pretrained-bert
```
For migration to new library, look [here](https://github.com/huggingface/transformers#migrating-from-pytorch-pretrained-bert-to-transformers). For baseline experiments, install `scikit-learn` as well.

## Data
**Raw** data can be found under `exps-data/data/*.txt` (this was provided by task organizers).

**Pre-preprocessed** data can be found under `exps-data/data/{train, dev, test}_data.pkl` as pickled files. English translations are also provided for reproducibility (Google Translate API was used to get translations). 

**ICD-10 Metadata** can be found under `exps-data/codes_and_titles_{de, en}.txt`, where each line is tab delimited as `[ICD Code Description] \t [ICD Code]`.

## Pre-trained Models

For **static word embeddings**, we used [English](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz) and [German](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.vec.gz) vectors provided by fastText. For domain specific vectors, we used [PubMed](https://archive.org/details/pubmed2018_w2v_400D.tar) word2vec (only for English).

For **contextualized word embeddings**, [BERT-base-cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip) and [BioBERT](https://github.com/naver/biobert-pretrained/releases/tag/v1.0-pubmed-pmc) for English and [Multilingual-BERT-base-cased](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip) for German.

Store all the models under a directory `MODELS`.

## Running BERT Models

Set the path `export BERT_MODEL=$MODELS/pubmed_pmc_470k` (e.g. BioBERT).

##### Convert TF checkpoint to PyTorch model
This script is provided by transformers library, but there might be some changes with new version so it is recommended to use the one installed with `pytorch-pretrained-bert`:

```bash
python convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path $BERT_MODEL/biobert_model.ckpt \
    --bert_config_file $BERT_MODEL/bert_config.json \
    --pytorch_dump_path $BERT_MODEL/pytorch_model.bin
```

##### Fine-tune the model

Configure the paths:
```bash
export DATA_DIR=exps-data/data
export BERT_EXPS_DIR=tmp/bert-exps-dir

```
Run the model:
```bash
python bert_multilabel_run_classifier.py \
    --data_dir $DATA_DIR \
    --use_data en \
    --bert_model $BERT_MODEL \
    --task_name clef \
    --output_dir $BERT_EXPS_DIR/output \
    --cache_dir $BERT_EXPS_DIR/cache \
    --max_seq_length 256 \
    --num_train_epochs 7.0 \
    --do_train \
    --do_eval \
    --train_batch_size 16
```

##### Inference

Run predictions (change files to test/dev manually in processor):

```bash
python bert_multilabel_run_classifier.py \
    --data_dir $DATA_DIR \
    --use_data en \
    --bert_model $BERT_EXPS_DIR/output \
    --task_name clef \
    --output_dir $BERT_EXPS_DIR/output \
    --cache_dir $BERT_EXPS_DIR/cache \
    --max_seq_length 256 \
    --do_eval 
```

##### Evaluate

Use official `evaluation.py` script to evaluate:

```bash
python evaluation.py --ids_file=$DATA_DIR/ids_test.txt \
                     --anns_file=$DATA_DIR/anns_test.txt \
                     --dev_file=$DATA_DIR/preds_test.txt \
                     --out_file=$DATA_DIR/eval_output.txt
```

## Running Other Models

Change configurations [here](https://github.com/suamin/multilabel-classification-bert-icd10/blob/master/main.py#L161) (no CLI yet).  Main parameters are:

`lang`: can be one of `{en, de}`

`load_pretrain_ft`: whether to use fastText pre-trained embeddings, works for both languages.

`load_pretrain_pubmed`: whether to use PubMed embeddings, works for English only.

`pretrain_file`: path to pre-trained vectors, should be one of `path/to/cc.{en, de}.300.vec` when `load_pretrain_ft=True` and `path/to/pubmed2018_w2v_400D.bin` when `load_pretrain_pubmed=True`.

`model_name`: name of the model; can be one of `{cnn, han, slstm, clstm}`.

For other hyperparameters, check [here](https://github.com/suamin/multilabel-classification-bert-icd10/blob/master/main.py#L46).

After all the models have been tested and results placed under one [directory](https://github.com/suamin/multilabel-classification-bert-icd10/blob/master/predict.py#L37) (one has to manually check the folder names), use `predict.py` to reproduce the numbers found in `Results.txt`.
