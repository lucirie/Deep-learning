import os
from PIL import Image
import numpy as np

def load_dataset(folder_path, image_size=(28, 28)):
    X = []
    y = []
    
    for label, class_name in enumerate(['no', 'yes']):
        class_folder = os.path.join(folder_path, class_name)
        
        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            
            # Load the image
            image = Image.open(file_path).convert('L')  # 'L' = grayscale
            image = image.resize(image_size)
            
            # Convert to numpy array and normalize to [0,1]
            image_array = np.array(image) / 255.0
            
            X.append(image_array)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y