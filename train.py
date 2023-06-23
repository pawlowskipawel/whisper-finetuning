from whisper_finetuning.training import WhisperTrainer, get_optimizer, freeze_encoder
from whisper_finetuning.data import WhisperDataset, DataCollatorCTCWithPadding
from whisper_finetuning.conf import parse_cfg

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

import random
import torch
import wandb
import os
import gc


def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

if __name__ == "__main__":
    seed_everything()
    cfg = parse_cfg()
        
    train_df = pd.read_csv(cfg.TRAIN_ANNOTATION_PATH)
    train_df["transcript"] = train_df["transcript"].str.replace(r" +", " ", regex=True)
    
    valid_df = pd.read_csv(cfg.VALID_ANNOTATION_PATH)
    valid_df["transcript"] = valid_df["transcript"].str.replace(r" +", " ", regex=True)
    
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{cfg.model_size}")
    
    best_results_dict = {}
    
    train_df["tokenized_transcript"] = train_df["transcript"].apply(lambda sentence: processor.tokenizer.encode(f"{sentence}<|endoftext|>", add_special_tokens=False))
    valid_df["tokenized_transcript"] = valid_df["transcript"].apply(lambda sentence: processor.tokenizer.encode(f"{sentence}<|endoftext|>", add_special_tokens=False))
    
    valid_df["len"] = valid_df["tokenized_transcript"].apply(len)
    valid_df = valid_df.sort_values("len")
    
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    train_dataset = WhisperDataset(train_df, processor, mel_dir=cfg.TRAIN_MEL_DIR)
    valid_dataset = WhisperDataset(valid_df, processor, mel_dir=cfg.VALID_MEL_DIR)
    
    data_collator = DataCollatorCTCWithPadding(processor)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, pin_memory=True, shuffle=True, drop_last=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, pin_memory=True, shuffle=False, collate_fn=data_collator)
    
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{cfg.model_size}")
    
    if cfg.freeze_encoder:
        freeze_encoder(model)
    
    model.to(cfg.device)
    
    optimizer = get_optimizer("adamw", model, learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_dataloader_len = len(train_dataloader)
    steps_per_epoch = (train_dataloader_len // cfg.grad_accum_steps) + (1 if train_dataloader_len % cfg.grad_accum_steps else 0)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.max_learning_rate,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=cfg.scheduler_warmup_epochs / cfg.epochs,
        div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor)
        
    trainer = WhisperTrainer(model, processor, optimizer, cfg, lr_scheduler=lr_scheduler)
    
    best_results = trainer.train(cfg.epochs, train_dataloader, valid_dataloader)
    
    for key, value in best_results.items():
        best_results_dict[key] = best_results_dict.get(key, []) + [value]
            
    del model, optimizer, trainer
    torch.cuda.empty_cache()
    gc.collect()

if cfg.wandb_log:
    for i, (key, values) in enumerate(best_results_dict.items()):
        wandb.run.summary[key] = np.array(values).mean()
