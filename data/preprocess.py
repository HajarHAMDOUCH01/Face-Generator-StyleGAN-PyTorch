import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import cv2
from multiprocessing import Pool, cpu_count
from functools import partial

def process_single_image_opencv(img_path, output_path, target_size):
    try:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        filename = img_path.stem + '.npy'
        np_path = output_path / filename
        np.save(np_path, img)
        return True
    except Exception as e:
        return False

def preprocess_ffhq_fast(input_root, output_root, target_size=128, limit=None, num_workers=None):
    input_path = Path(input_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = []
    for img_path in sorted(input_path.rglob("*.png")):
        images.append(img_path)
        if limit and len(images) >= limit:
            break
    
    num_workers = num_workers or cpu_count()
    print(f"Found {len(images)} images to preprocess")
    print(f"Using {num_workers} parallel workers with OpenCV")
    
    process_func = partial(
        process_single_image_opencv,
        output_path=output_path,
        target_size=target_size
    )
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, images, chunksize=50),
            total=len(images),
            desc="Converting to numpy"
        ))
    
    print(f"\nComplete! Processed {sum(results)}/{len(images)} images")
