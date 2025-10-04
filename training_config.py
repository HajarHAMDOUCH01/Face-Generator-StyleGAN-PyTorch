training_config = {
    # Architecture
    "image_size": 128,
    "z_dim": 512,
    "w_dim": 512,
    "mapping_layers": 4,  
    
    # Training
    "batch_size": 32,
    "num_epochs": 300,
    "num_workers": 2,

    # Learning rates
    "g_lr": 0.004,      
    "d_lr": 0.0002,     
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,
    
    # Regularization
    "r1_gamma": 0.5,    
    "mixing_prob": 0.9,  
    "style_mixing_prob": 0.9,  
    
    # Loss
    "loss_type": "non_saturating",
    
    "r1_interval": 16,  
    "n_critic" : 2,
    
    # Progressive growing
    "start_size": 4,
    "progressive_growing": False,
    
    # Paths
    "dataset_path": "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
    "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
    
    "save_every": 2,
    "sample_every": 5,
    "log_every": 50,
    
    # Gradient clipping 
    "grad_clip": 1.0,  
}