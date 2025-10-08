training_config = {
    "image_size": 128,
    "z_dim": 512,
    "w_dim": 512,
    "mapping_layers": 8,  
    "dataset_limit" : 10000,
    
    "batch_size": 52,  
    "num_epochs": 20,
    "num_workers": 2,
   
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,


    
    "g_lr": 0.0005,     
    "d_lr": 0.002,      
    
    "plr_weight": 1.0,   
    "plr_interval": 16,  
    
    "r1_gamma": 8.0,    
    "r1_interval": 16,  

    "style_mixing_prob": 0.9,  
        
    "dataset_path": "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
    "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
    
    "save_every": 2,
    "sample_every": 2,
    "log_every": 50,

    "plr_decay": 0.01,      

    # ADA config
    "target_rt":0.6,
    "adjustment_speed_imgs":500_000,
    "initial_p":0.0,
    "update_interval":4,        
}