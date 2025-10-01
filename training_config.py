training_config = {
    # Architecture
    "image_size": 128,  # Target resolution
    "z_dim": 512,  # Latent dimension
    "w_dim": 512,  # Intermediate latent dimension
    "mapping_layers": 8,  # Depth of mapping network (paper uses 8)
    
    # Training
    "batch_size": 16,
    "num_epochs": 300,
    "num_workers": 2,
    
    "g_lr": 0.002,  # Generator learning rate
    "d_lr": 0.002,  # Discriminator learning rate
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,
    
    # Regularization
    "r1_gamma": 10.0,  # R1 regularization weight (paper uses 10)
    "mixing_prob": 0.9,  # Style mixing probability (paper uses 90%)
    "style_mixing_prob": 0.9,  
    
    # Loss
    "loss_type": "non_saturating",  
    
    # Progressive growing settings
    "start_size": 4,  # Starting resolution (paper starts from 4x4)
    "progressive_growing": False,  
    
    # Paths
    "dataset_path": "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
    "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
    
    "save_every": 10,
    "sample_every": 5,
    "log_every": 50,  
    
    # Gradient clipping
    "grad_clip": None,  
}