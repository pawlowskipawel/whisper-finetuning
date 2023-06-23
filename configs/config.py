from intron_afrispeech.metrics import WER

# semicolon + colon

args = {
    # glogal
    "fp16": False,
    "TRAIN_ANNOTATION_PATH": "data/train_metadata.csv",
    "TEST_ANNOTATION_PATH": "data/valid_metadata.csv",
    "VALID_ANNOTATION_PATH": "data/valid_metadata.csv",
    
    "TRAIN_MEL_DIR": "data/train_mels",
    "VALID_MEL_DIR": "data/valid_mels",
    "TEST_MEL_DIR": "data/valid_mels",

    "device": "cuda",
    "save_path": "checkpoints",

    # model
    "freeze_encoder": True,
    "model_size": "tiny",
    
    # training
    "epochs": 5,
    "batch_size": 24,
    "grad_accum_steps": 1, # effective batch size = batch_size * grad_accum_steps
    "grad_clip_norm": 1.0,
    "first_eval_epoch": 1,
    "validation_step": 1000,

    # optimizer
    "learning_rate": 6e-5,
    "weight_decay": 1e-4,
    
    # lr scheduler
    "scheduler_warmup_epochs": 0,
    "max_learning_rate": 6e-5,
    "div_factor": 1.0,
    "final_div_factor": 10.0,
}

args["metrics_dict"] = {
    "WER": WER()
}