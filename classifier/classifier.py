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

image_rescale = transforms.Compose([SquarePad(padding_mode="symmetric"),transforms.Resize((512,512))])
image_rescale2 = transforms.v2.RandomCrop(512,pad_if_needed = True,padding_mode='symmetric')

class Resnet_block(torch.nn.Module):
    def __init__(self,channels_in,channels,reduction = "this is just so both blocks have same input",stride = 1,layers = 2):
        super(Resnet_block,self).__init__()
        self.stride = stride
        #the residual block
        #manditory first layer
        if layers < 1:
            raise ValueError("Resnet_block needs at least 1 layer, has " + str(layers))
        block_layers = [
            torch.nn.Conv2d(channels_in,channels,3,stride,1,padding_mode='zeros', bias=False),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU()
        ]
        #additional layers (allows custom numbers of layers)
        add_layers = layers - 1
        for i in range(add_layers):
            block_layers.append(torch.nn.Conv2d(channels,channels,3,1,1,padding_mode='zeros', bias=False))
            block_layers.append(torch.nn.BatchNorm2d(channels))
        if i != add_layers - 1: #no relu for last layer
            block_layers.append(torch.nn.ReLU())
    
        self.main = torch.nn.Sequential(*block_layers)#unpack list into sequential
        self.resid = torch.nn.Conv2d(channels_in,channels,1,stride,1) #the residual layer the input is fed through
        self.relu = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm2d(channels)

    def forward(self,inp):
        x = self.main(inp)
        x += (inp if self.stride == 1 else self.resid(inp))#self.resid(inp) #apply residual
        x = self.norm(x)
        x = self.relu(x) #the missing activation from the residual block
        return x

class Resnet_bottle_block(torch.nn.Module):
    def __init__(self,channels_in,channels,reduction,stride = 1):
        super(Resnet_block,self).__init__()
        self.stride = stride
        #the residual block
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in,channels_in/reduction,1,1,1,padding_mode='zeros', bias=False),
            torch.nn.BatchNorm2d(channels_in/reduction),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels_in/reduction,channels/reduction,3,stride,1,padding_mode='zeros', bias=False),
            torch.nn.BatchNorm2d(channels/reduction),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels/reduction,channels,1,1,1,padding_mode='zeros', bias=False),
            torch.nn.BatchNorm2d(channels)
        )
        self.resid = torch.nn.Conv2d(channels_in,channels,1,stride,1) #the residual layer the input is fed through
        self.relu = torch.nn.ReLU()
        self.norm = torch.nn.BatchNorm2d(channels)

    def forward(self,inp):
        x = self.main(inp)
        x += (inp if self.stride == 1 else self.resid(inp))#self.resid(inp) #apply residual
        x = self.norm(x)
        x = self.relu(x) #the missing activation from the residual block
        return x

class Resnetish(torch.nn.Module):
    def __init__(self,num_classes,image_size,stages,block,starting_channels = 32,reduction = 4):#reduction only for bottleblocks
        super(Resnetish,self).__init__()
        reduction = 2*2*(2**len(stages - 1))
        layers = []
        layers.append(torch.nn.Conv2d(3,starting_channels,7,3,2,padding_mode='zeros', bias=False))
        image_size /=2
        layers.append(torch.nn.BatchNorm2d(starting_channels))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.MaxPool2d(3,2,1))
        image_size/=2
        cur_channels = starting_channels
        for i in range(stages[0]):
            layers.append(block(cur_channels,cur_channels,reduction))
        for stage in stages[1:]:
            next_channels = cur_channels*2
            layers.append(block(cur_channels,next_channels,reduction,2))#strided block
            image_size/=2
            for i in range(stage - 1):
                layers.append(block(next_channels,next_channels,reduction))
            cur_channels = next_channels
            
        layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        cur_channels = 1#thats what that pooling does?
        layers.append(torch.nn.Flatten())
        outsize = cur_channels*image_size*image_size
        layers.append(torch.nn.Linear(outsize,num_classes))
        layers.append(torch.nn.Softmax())

        self.main = torch.nn.Sequential(*layers)
        # self.main = torch.nn.Sequential( #Bx3x512x512
        #     pool_norm(2,48), #Bx48x256x256
        #     pool_norm(2,96), #Bx96x128x128
        #     pool_norm(2,192), #Bx192x64x64
        #     pool_norm(2,384), #Bx384x32x32
        #     pool_norm(2,768), #Bx768x16x16
        #     pool_norm(2,1536), #Bx1536x8x8
        #     pool_norm(2,3072), #Bx3072x4x4
        #     pool_norm(2,6144), #Bx6144x2x2
        #     pool_norm(2,12288), #Bx12288x1x1
        #     torch.nn.Flatten(), #Bx12288
        #     torch.nn.Linear(12288,num_classes), #Bx128
        #     torch.nn.ReLU(),
        #     torch.nn.Softmax()
        # )
        
    def forward(self,inp):
         return self.main(inp)

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