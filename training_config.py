training_config = {
    "image_input_size": 128,  # Target resolution (4 -> 8 -> 16 -> 32 -> 64 -> 128)
    
    "z_dim": 512,  
    "batch_size": 32,  
    "lr": 0.001,  
    "num_epochs": 300,
    "num_workers": 2,  

    "beta_start": 0.0,  
    "beta_end": 0.5,  
    "beta_warmup_epochs": 50,  
    
    "mse_weight": 1.0, 
    "percep_weight": 0.1, 
    "perceptual_start_epoch": 10,  
    
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-8,
    
    "dataset_path": "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
    "save_dir": "/content/drive/MyDrive",
    
    "save_every": 10,  
    "sample_every": 5,  
}