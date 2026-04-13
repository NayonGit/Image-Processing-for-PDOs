import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

TELLU_PATH = 'data/tellu.h5'
ORGAQUANT_PATH = 'data/orgaquant.h5'
MULTIORG_PATH = 'data/multiorg.h5'

def load_and_show_example(img_index=0, data_path=TELLU_PATH):
    with h5py.File(data_path, 'r') as hdf:
        group = hdf['train']
        img_names = list(group['images'].keys())
        
        target_name = img_names[img_index]
        print(f"Loading the image : {target_name}")

        img = np.array(group['images'][target_name])
        labels = np.array(group['labels'][target_name])

    # Plots
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    
    # Bounding Boxes
    for lbl in labels:
        # lbl = [class, x1, y1, x2, y2]
        if data_path == TELLU_PATH:
            x1, y1, x2, y2 = lbl[1], lbl[2], lbl[3], lbl[4]
        elif data_path == ORGAQUANT_PATH:
            x1, y1, x2, y2 = lbl[0], lbl[1], lbl[2], lbl[3]
        elif data_path == MULTIORG_PATH:
            x1, y1, x2, y2 = lbl[1], lbl[2], lbl[3], lbl[4]
        else:
            raise ValueError("Unknown data path")
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"Org", color='white', bbox=dict(facecolor='red', alpha=0.5))

    plt.title(f"Visualization : {target_name}")
    plt.axis('off')
    plt.show()