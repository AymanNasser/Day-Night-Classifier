import pandas as pd
import os

def get_train_df(root_dir):
    images = []
    labels = [] 

    for category in os.listdir(root_dir):
        
        if category == 'day':
            for file in os.listdir(os.path.join(root_dir, category)):
                images.append(file)
                labels.append(0)
            
        elif category == 'night':
            for file in os.listdir(os.path.join(root_dir, category)):
                images.append(file)
                labels.append(1)

        else:
            raise OSError("Non Supported Category")

    data = {'Images': images, 'labels': labels}
    df = pd.DataFrame()
    
    return df
