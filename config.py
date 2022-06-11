import os


class Config:
    # Data
    image_size = 224
    input_dim = 3
    num_classes = 4

    # Trainer
    train_batch_size = 32
    val_batch_size = 128
    num_workers = 4
    epochs = 40
    lr = 1e-3
    seed = 42
