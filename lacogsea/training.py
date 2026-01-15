from __future__ import annotations
import torch
import logging
from torch.cuda.amp import autocast
import torch.nn as nn

def train_auto_encoder(encoder, decoder, dataloader, criterion, optimizer, scaler, device,
                       regularization=None, l1_regular_lambda=1e-5, l2_regular_lambda=1e-5):
    import warnings
    
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_samples = 0

    for bulk, *_ in dataloader:
        bulk = bulk.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with autocast():
                encoded = encoder(bulk)
                reconstructed = decoder(encoded)

                loss = criterion(reconstructed, bulk)

                if regularization == 'l1':
                    l1_norm_enc = sum(p.abs().sum() for p in encoder.parameters())
                    l1_norm_dec = sum(p.abs().sum() for p in decoder.parameters())
                    loss += l1_regular_lambda * (l1_norm_enc + l1_norm_dec)

                elif regularization == 'l2':
                    l2_norm_enc = sum(p.pow(2).sum() for p in encoder.parameters())
                    l2_norm_dec = sum(p.pow(2).sum() for p in decoder.parameters())
                    loss += l2_regular_lambda * (l2_norm_enc + l2_norm_dec)

                elif regularization == 'elastic':
                    l1_norm_enc = sum(p.abs().sum() for p in encoder.parameters())
                    l2_norm_enc = sum(p.pow(2).sum() for p in encoder.parameters())
                    l1_norm_dec = sum(p.abs().sum() for p in decoder.parameters())
                    l2_norm_dec = sum(p.pow(2).sum() for p in decoder.parameters())
                    loss += l1_regular_lambda * (l1_norm_enc + l1_norm_dec) + \
                            l2_regular_lambda * (l2_norm_enc + l2_norm_dec)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * bulk.size(0)
        total_samples += bulk.size(0)

    return total_loss / total_samples
