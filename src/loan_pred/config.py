import torch.cuda

VARS = {
    "emp_status_mode": "Permanent"
}

config = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "TRAIN_BS": 16,
    "VALID_BS": 250,
    "RANDOM_STATE": 42,
    "LR": 0.01,
    "CUT_POINT": 0.5,
    "CHECKPOINT_PATH": "../../models_storage/embeddings/checkpoint.pt",
    "N_EPOCHS": 100,
    "DROPOUT": 0.4,
}
