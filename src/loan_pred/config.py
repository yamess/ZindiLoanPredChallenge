import torch.cuda

VARS = {
    "emp_status_mode": "Permanent"
}

config = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "TRAIN_BS": 32,
    "VALID_BS": 100,
    "RANDOM_STATE": 42,
    "LR": 0.1,
    "CUT_POINT": 0.5,
    "CHECKPOINT_PATH": "../../models_storage/embeddings/checkpoint.pt",
    "N_EPOCHS": 500,
    "cut_point": 0.5
}
