training_config = {
    # Architecture
    "image_size": 128,
    "z_dim": 512,
    "w_dim": 512,
    "mapping_layers": 8,  
    "dataset_limit": 60000,  
    
    # Training
    "batch_size": 32,  
    "num_epochs": 100, 
    "num_workers": 2,  
   
    # Optimizer - StyleGAN2 standard settings
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,
    
    # Learning rates 
    "g_lr": 0.0025,  
    "d_lr": 0.0025,  
    
    # Path Length Regularization
    "plr_weight": 2.0,  
    "plr_interval": 4, 
    "plr_decay": 0.01,
    
    # R1 Regularization (critical for stability)
    "r1_gamma": 1.0,  
    "r1_interval": 16,  
    
    # Style Mixing
    "style_mixing_prob": 0.9,  
        
    # Paths
    "dataset_path": "/content/drive/MyDrive/ffhq-dataset/images1024x1024",  
    "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
    
    # Logging
    "save_every": 5,  
    "sample_every": 1, 
    "log_every": 100, 

    # ADA Configuration (optimized for FFHQ 70k)
    "target_rt": 0.6,  
    "adjustment_speed_imgs": 500_000,  
    "initial_p": 0.0,  
    "update_interval": 4,  
}