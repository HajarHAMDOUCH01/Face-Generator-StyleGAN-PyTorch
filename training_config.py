training_config = {
    "image_size": 128,
    "z_dim": 512,
    "w_dim": 512,
    "mapping_layers": 8,  
    
    "batch_size": 50, 
    "num_epochs": 300,
    "num_workers": 4,

    "g_lr": 0.002,       
    "d_lr": 0.0005,       
    "adam_beta1": 0.0,
    "adam_beta2": 0.99,
    "adam_eps": 1e-6,
    
    "style_mixing_prob": 0.9,  
        
    "dataset_path": "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba",
    "save_dir": "/content/drive/MyDrive/stylegan_checkpoints",
    
    "save_every": 2,
    "sample_every": 2,
    "log_every": 50,

    "plr_weight": 2.0,  
    "plr_interval": 4,   
    "plr_decay": 0.01,      
    
    "r1_gamma": 15.0,
    "r1_interval": 4,  

}

"""
If discriminator dominates (real_score >> 0, fake_score << 0):

Lower r1_gamma to 5.0
Increase d_lr relative to g_lr


If generator dominates (both scores near 0):

Increase r1_gamma to 15.0
Check if images look decent despite scores


If training is unstable:

Reduce both learning rates to 0.001
Increase r1_interval to 16
"""