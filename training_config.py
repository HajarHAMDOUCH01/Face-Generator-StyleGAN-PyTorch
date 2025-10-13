training_config = {
    # Architecture
    "image_size": 128,
    "z_dim": 512,
    "w_dim": 512,
    "mapping_layers": 8,
    "style_mixing_prob": 0.9,
    
    # Dataset
    "dataset_path": "/content/drive/MyDrive/ffhq-dataset/images1024x1024",  # images
    "preprocess_data": False,  
    "processed_dataset_path": "/content/drive/MyDrive/stylegan_processed_dataset",  # Preprocessed as numpy arrays 
    "using_dat_file" : True,
    "memmap_path": "/content/drive/MyDrive/ffhq_memmap.dat",
    "dataset_limit": 70000,
    
    # Training
    "batch_size": 32,
    "num_epochs": 100,
    "num_workers": 8,
    "prefetch_factor": 4,
    
    # Optimizer
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,
    "g_lr": 0.002,
    "d_lr": 0.002,
    
    # Regularization
    "plr_weight": 2.0,
    "plr_interval": 4,
    "plr_decay": 0.01,
    "r1_gamma": 1.0,
    "r1_interval": 16,
    "warmup_epochs": 5,
    
    # Logging
    "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
    "save_every": 1,
    "sample_every": 1,
    "log_every": 100,
    
    "use_hf_upload": True,  
}
