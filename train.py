# -*- coding: utf-8 -*-

import torch
import numpy as np

from torch import optim
from loss import BalancedBCEWithLogitsLoss
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score
from tqdm import tqdm, trange


def evaluate(dataloader, model, device, no_labels=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    logits = []
    preds = []
    labels = []
    ids = []
    avg_loss = 0.
    loss_fct = BCEWithLogitsLoss()
    
    def append(all_tensors, batch_tensor):
        if len(all_tensors) == 0:
            all_tensors.append(batch_tensor)
        else:
            all_tensors[0] = np.append(all_tensors[0], batch_tensor, axis=0)
        return all_tensors
    
    def detach(tensor, dtype=None):
        if dtype:
            return tensor.detach().cpu().numpy().astype(dtype)
        else:
            return tensor.detach().cpu().numpy()
    
    with torch.no_grad():  
        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(device) for t in batch)
            if no_labels:
                b_inputs, b_ids = batch
            else:
                b_inputs, b_labels, b_ids = batch

            b_logits = model(b_inputs)
            if not no_labels:
                loss = loss_fct(b_logits, b_labels)
                avg_loss += loss.item()
            
            b_preds = (torch.sigmoid(b_logits).detach().cpu().numpy() >= 0.5).astype(int)
            b_logits = detach(b_logits, float)
            if not no_labels:
                b_labels = detach(b_labels, int)
            b_ids = detach(b_ids, int)
            
            preds = append(preds, b_preds)
            logits = append(logits, b_logits)
            if not no_labels:
                labels = append(labels, b_labels)
            ids = append(ids, b_ids)
    
    preds = preds[0]
    logits = logits[0]
    if not no_labels:
        labels = labels[0]
        avg_loss /= len(dataloader)
    ids = ids[0]
    
    if not no_labels:
        score = f1_score(y_true=labels, y_pred=preds, average='micro')
        print("\nEvaluation - loss: {:.6f}  f1: {:.4f}%\n".format(avg_loss, score))
    else:
        score = 0.
    
    return score, (logits, preds, labels, ids, avg_loss)


def train(train_dataloader, dev_dataloader, model, 
          epochs, lr, device, grad_clip=None, 
          model_save_fname="model.pt"):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    steps = 0
    best_fmicro = None
    last_fmicro = None
    loss_fct = BCEWithLogitsLoss()
    evals = []
    
    try:
        for epoch_no in trange(epochs, desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                inputs, labels, ids = batch
                
                logits = model(inputs)
                loss = loss_fct(logits, labels)
                loss.backward()
                
                tr_loss += loss.item()
                nb_tr_examples += inputs.size(0)
                nb_tr_steps += 1
                
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                steps += 1 
            
            score, eval_data = evaluate(dev_dataloader, model, device)
            
            if not best_fmicro or score < best_fmicro:
                torch.save(model.state_dict(), "./{}".format(model_save_fname))
                best_fmicro = score
            
            evals.append((epoch_no, score, eval_data))
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    
    return evals
