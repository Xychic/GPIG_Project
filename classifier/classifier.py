import torch
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from dataset_filter.dataset_filter import getContent, binIndex

class TreeSpeciesDataset(torch.utils.data.Dataset):
    def __init__(self, tree_dir, species_list, transform=None, target_transform=None):
        self.tree_dir = tree_dir
        self.transform = transform
        self.target_transform = target_transform
        #create species to id map
        self.species_dict = dict()
        for i in range(len(species_list)):
            self.species_dict[species_list[i]] = i
        #get and store image id pairs
        self.records = []
        for file in os.listdir(tree_dir):
            if file[-4:] == ".xml":
                species = getContent(self.tree_dir,file)[1]
                if species is not None:#seems to be a valid xml file, a species tag was found
                    if os.path.isfile(os.path.join(self.tree_dir,file[:-4] + ".png")):
                        self.records.append((file[:-4] + ".png",self.species_dict[species]))
                    elif os.path.isfile(os.path.join(self.tree_dir,file[:-4] + ".jpg")):
                        self.records.append((file[:-4] + ".jpg",self.species_dict[species]))
        
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img_path = os.path.join(self.tree_dir, record[0])
        image = read_image(img_path)
        label = record[1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
