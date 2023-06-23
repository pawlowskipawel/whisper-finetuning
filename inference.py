from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_finetuning.data import WhisperDataset, DataCollatorCTCWithPadding
from whisper_finetuning.conf import parse_cfg
from torch.utils.data import DataLoader

from dataclasses import dataclass
from tqdm import tqdm

import pandas as pd
import numpy as np

import random
import torch
import os
import re

def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def process_batch(batch, device="cuda"):
    input_features = batch["input_features"].to(device)
    decoder_input_ids = batch["decoder_input_ids"].to(device)

    return input_features, decoder_input_ids


if __name__ == "__main__":
    seed_everything()
    cfg = parse_cfg()
    
    best_results_dict = {}
    df = pd.read_csv(cfg.TEST_ANNOTATION_PATH)

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{cfg.model_size}")
    dataset = WhisperDataset(df, processor, mel_dir=cfg.TEST_MEL_DIR, mode="inference")
    
    data_collator = DataCollatorCTCWithPadding(processor, training=False)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, pin_memory=True, shuffle=False, collate_fn=data_collator)
    
    model = WhisperForConditionalGeneration.from_pretrained(os.path.join(cfg.save_path, cfg.config_name, "last"))
    model.to(cfg.device)
    model.eval()
    
    full_predictions = []
    
    with tqdm(dataloader, unit="batch", bar_format='{l_bar}{bar:10}{r_bar}', total=len(dataloader)) as progress_bar, torch.cuda.amp.autocast(enabled=cfg.fp16):
        progress_bar.set_description(f"Inference".ljust(25))
        
        for batch in progress_bar:
            input_features, decoder_input_ids = process_batch(batch, device=cfg.device)
            
            predictions = model.generate(input_features=input_features, decoder_input_ids=decoder_input_ids, max_new_tokens=100)

            predictions = processor.batch_decode(predictions, skip_special_tokens=True)
            
            full_predictions.extend(predictions)


df["pred_transcript"] = full_predictions
df["pred_transcript"] = df["transcript"].str.replace(" +", " ", regex=True).str.lower()
df.to_csv(f"{cfg.config_name}.csv", index=None)
    
        

        
