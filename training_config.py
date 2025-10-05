training_config = {
    "image_size": 128,
    "z_dim": 512,
    "w_dim": 512,
    "mapping_layers": 8,  
    
    "batch_size": 50, 
    "num_epochs": 300,
    "num_workers": 2,

    "g_lr": 0.0015,       
    "d_lr": 0.0005,       
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,
    
    "style_mixing_prob": 0.9,  
    
    "r1_interval": 16,  
    
    "dataset_path": "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
    "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
    
    "save_every": 2,
    "sample_every": 2,
    "log_every": 50,

    "plr_weight": 2.0,  
    "plr_interval": 16,   
    "plr_decay": 0.01,      
    
    "r1_gamma": 10.0,
}