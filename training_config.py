training_config = {
    # Architecture
    "image_size": 128,
    "z_dim": 512,
    "w_dim": 512,
    "mapping_layers": 8,
    
    # Dataset
    "dataset_path": "/content/ffhq_128x128_jpg",  
    "dataset_limit": 70000,
    
    # Training
    "batch_size": 40,
    "num_epochs": 100,
    "num_workers": 8,
    "prefetch_factor": 4,
    "style_mixing_prob": 0.8,
    
    # Optimizer
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,
    "g_lr": 0.00025,
    "d_lr": 0.00020,
    
    # Regularization
    "plr_weight": 2.0,
    "plr_interval": 4,
    "plr_decay": 0.01,
    "r1_gamma": 5.0,
    "r1_interval": 16,
    "warmup_epochs": 5,
    
    # Logging
    "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
    "save_every": 1,
    "sample_every": 1,
    "log_every": 100,
    
    "use_hf_upload": True,  
}





# training_config = {
#     # Architecture
#     "image_size": 128,
#     "z_dim": 512,
#     "w_dim": 512,
#     "mapping_layers": 8,
#     "style_mixing_prob": 0.75,
    
#     # Dataset
#     "dataset_path": "/content/ffhq_128x128_jpg",  
#     "dataset_limit": 70000,
    
#     # Training
#     "batch_size": 40,
#     "num_epochs": 100,
#     "num_workers": 8,
#     "prefetch_factor": 4,
    
#     # Optimizer
#     "adam_beta1": 0.0,
#     "adam_beta2": 0.99,
#     "adam_eps": 1e-8,
#     "g_lr": 0.00017,
#     "d_lr": 0.00012,
    
#     # Regularization
#     "plr_weight": 1.0,
#     "plr_interval": 4,
#     "plr_decay": 0.01,
#     "r1_gamma": 8.0,
#     "r1_interval": 16,
#     "warmup_epochs": 5,
    
#     # Logging
#     "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
#     "save_every": 1,
#     "sample_every": 1,
#     "log_every": 100,
    
#     "use_hf_upload": True,  
# }














# training_config = {
#     "image_size": 128,
#     "z_dim": 512,
#     "w_dim": 512,
#     "mapping_layers": 8,
#     "style_mixing_prob": 0.20, 
    
#     "dataset_path": "/content/ffhq_128x128_jpg",  
#     "dataset_limit": 70000,
    
#     "batch_size": 32,
#     "num_epochs": 100,
#     "num_workers": 8,
#     "prefetch_factor": 4,
    
#     "adam_beta1": 0.0,
#     "adam_beta2": 0.99,
#     "adam_eps": 1e-8,
#     "g_lr": 0.00007,   
#     "d_lr": 0.00012,   
    
#     "plr_weight": 1.0,      
#     "plr_interval": 6,
#     "plr_decay": 0.01,
#     "r1_gamma": 8.5,      
#     "r1_interval": 16,
#     "warmup_epochs": 0,
    
#     "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
#     "save_every": 1,
#     "sample_every": 1,
#     "log_every": 100,
    
#     "use_hf_upload": True,

# }