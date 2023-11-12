import torch
from torchvision.io import read_image
from torchvision import transforms 
from sklearn.model_selection import train_test_split
import os
import sys
import math
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

def safe_train_test_split(dataset,test_size): #"adapted" from eric's answer on https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.targets,
        stratify=dataset.targets,
        test_size=test_size
    )

    # generate subset based on indices
    train_split = torch.utils.data.Subset(dataset, train_indices)
    test_split = torch.utils.data.Subset(dataset, test_indices)
    return train_split,test_split

class SquarePad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def get_padding(self,img):
        h, w = img.size()[-2:]
        max_wh = max(w, h)
        h_padding = (max_wh - w) / 2
        v_padding = (max_wh - h) / 2
        return math.floor(h_padding), math.floor(v_padding), math.ceil(h_padding), math.ceil(v_padding)
        
    def __call__(self, img):
        return transforms.functional.pad(img, self.get_padding(img), self.fill, self.padding_mode)

image_rescale = transforms.Compose([SquarePad(padding_mode="reflect"),transforms.Resize((512,512))])

with open(os.path.abspath("dataset_filter\\listSpecies.txt")) as f:
    species_list = f.read().splitlines()

dat = TreeSpeciesDataset(os.path.abspath("..\\trees"),species_list,image_rescale)
max_width = 0
max_height = 0
min_width = 10000
min_height = 10000
species_dist = dict()
for i in range(len(species_list)):
    species_dist[i] = 0
for i in range(len(dat)):
    img,lab = dat[i]
    size = img.size()
    max_width = max(max_width,size[2])
    max_height = max(max_height,size[1])
    min_width = min(min_width,size[2])
    min_height = min(min_height,size[1])
    species_dist[lab] += 1
print((max_width,max_height))
print((min_width,min_height))
print((min(species_dist.values()),max(species_dist.values())))
for key, value in species_dist.items():
    if value < 5:
        print((species_list[key],value))