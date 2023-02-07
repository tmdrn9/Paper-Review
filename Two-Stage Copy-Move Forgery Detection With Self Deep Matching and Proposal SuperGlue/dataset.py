from torch.utils.data import Dataset
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, annotation, path, transforms=None):
        self.annotation = annotation
        self.img_names = annotation[['dup_img_name']].values
        self.dup_mask_names = annotation[['dup_mask_name']].values
        self.path = path
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.path + self.img_names[index][0]
        mask_path = self.path + self.dup_mask_names[index][0]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 127, 255, mask)
        mask = np.where(mask <= 127, 0, mask)
        mask = np.expand_dims(mask,axis=2)
        

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
              
        
        image=image.astype(np.float32).transpose(2,0,1)
        mask=mask.astype(np.float32).transpose(2,0,1)
        
        image/=255
        mask/=255 
            
        return image, mask

    def __len__(self):
        return len(self.img_names)