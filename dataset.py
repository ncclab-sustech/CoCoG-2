import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchtyping import TensorType
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import json
import os.path
import torch.utils.data as data

import sys
sys.path.append('/home/weichen/projects/visobj/related-projects/PerceptualSimilarity')
all_data_path='/home/weichen/projects/visobj/proposals/mise/data/ulgn/all_data.pt'

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        
    def name(self):
        return 'BaseDataset'
    
    def initialize(self):
        pass

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NP_EXTENSIONS = ['.npy',]

def is_image_file(filename, mode='img'):
    if(mode=='img'):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif(mode=='np'):
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)

def make_dataset(dirs, mode='img'):
    if(not isinstance(dirs,list)):
        dirs = [dirs,]

    images = []
    for dir in dirs:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    images.append(path)

    # print("Found %i images in %s"%(len(images),root))
    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def get_preprocess_fn(preprocess, load_size, interpolation):
    if preprocess == "LPIPS":
        t = transforms.ToTensor()
        return lambda pil_img: t(pil_img.convert("RGB")) / 0.5 - 1.
    else:
        if preprocess == "DEFAULT":
            t = transforms.Compose([
                transforms.Resize((load_size, load_size), interpolation=interpolation),
                transforms.ToTensor()
            ])
        elif preprocess == "DISTS":
            t = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        elif preprocess == "SSIM" or preprocess == "PSNR":
            t = transforms.ToTensor()
        else:
            raise ValueError("Unknown preprocessing method")
        return lambda pil_img: t(pil_img.convert("RGB"))
    

# triplet
class TripletData(Dataset):
    def __init__(self, triplets, n_objects: int):
        super(TripletData, self).__init__()
        if isinstance(triplets, torch.Tensor):
            self.triplets = triplets
        elif isinstance(triplets, np.ndarray):
            self.triplets = torch.from_numpy(triplets).type(torch.LongTensor)
        else:
            raise TypeError(f'\nData has incorrect type:{type(triplets)}\n')
        self.identity = torch.eye(n_objects).to(self.triplets.device)
    
    def encode_as_onehot(self, triplet: TensorType["k"]) -> TensorType["k", "m"]:
        """Encode a tensor of three numerical indices as a tensor of three one-hot-vectors."""
        return self.identity[triplet, :]

    def __getitem__(self, index: int) -> TensorType["k", "m"]:
        index_triplet = self.triplets[index]
        one_hot_triplet = self.encode_as_onehot(index_triplet)
        return one_hot_triplet

    def __len__(self) -> int:
        return self.triplets.shape[0]
    
class FoodDataset(Dataset):
    def __init__(self, directory, size=(224, 224), transform=None):
        self.directory = directory
        self.images = sorted([file for file in os.listdir(directory) if file.endswith('.png')])

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),  
                transforms.ToTensor()  
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        return image

class MultiDataset(Dataset):
    def __init__(self, directory, order, size=(224, 224), transform=None):
        self.directory = directory
        self.transform = transform
        self.order = order

        # List all jpg files
        all_images = [file for file in os.listdir(directory) if file.endswith('.jpg')]
        all_images.sort(key=self.sort_numeric)         

        # Sort images based on the order tensor
        self.sorted_images = [all_images[i] for i in order]
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),  
                transforms.ToTensor()  
            ])

    def __len__(self):
        return len(self.sorted_images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.sorted_images[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        return image
    
    def sort_numeric(self, file):
        return int(file.split('.')[0])

# image
class ImageData(Dataset):
    """
    path: Path to the directory where the images are saved. 
        Images are assumed to be saved in the format <label_name>/image.jpg 
        (e.g., zucchini/zucchini_08n.jpg)
    transform: A torchvision.transforms object that will be applied to the images
    """

    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.images = []
        self.labels = []
        self.categories = []

        # Look through each sub-directory in the path
        for label in os.listdir(self.path):
            for image in os.listdir(os.path.join(self.path, label)):
                self.images.append(os.path.join(self.path, label, image))
                self.labels.append(label)
            self.categories.append(label)

        self.categories = sorted(self.categories)
        self.label_to_index = {self.categories[i]: i for i in range(len(self.categories))}

    def __len__(self):
        # Return the total number of images
        return len(self.images)

    def __getitem__(self, idx):
        # Open and send one image and its label
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
           image = transforms.ToTensor()(image)
            
        label = self.label_to_index[self.labels[idx]]
        return image, label
    
def read_properties(tsv_path):
    # Read the TSV file into a DataFrame with 'uniqueID' as the index
    properties_df = pd.read_csv(tsv_path, sep='\t', index_col='uniqueID')
    # Convert the DataFrame into a dictionary with 'uniqueID' as keys
    properties_dict = properties_df.to_dict(orient='index')
    return properties_dict

class ImageData_P(Dataset):
    def __init__(self, path, transform=None, properties_path=None, target_property=None, subset=None, split_ratio=0.8):
        super().__init__()
        self.path = path
        self.transform = transform
        self.images = []
        self.labels = []
        self.categories = []
        self.properties = {}
        self.target_property = target_property 

        if properties_path is not None:
            self.properties = read_properties(properties_path)

        # Look through each sub-directory in the path
        for label in sorted(os.listdir(self.path)):
            image_paths = sorted(os.listdir(os.path.join(self.path, label)))
            for image in image_paths:
                self.images.append(os.path.join(self.path, label, image))
                self.labels.append(label)
            self.categories.append(label)

        self.label_to_index = {self.categories[i]: i for i in range(len(self.categories))}

        # Split the dataset if a subset is specified
        if subset in ['train', 'val']:
            total_images = len(self.images)
            split_idx = int(split_ratio * total_images)
            if subset == 'train':
                self.images = self.images[:split_idx]
                self.labels = self.labels[:split_idx]
            elif subset == 'val':
                self.images = self.images[split_idx:]
                self.labels = self.labels[split_idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        label = self.labels[idx]
        label_index = self.label_to_index[label]

        # Retrieve the target property score for the given label if it exists
        target_score = None
        if self.target_property and label in self.properties:
            target_score = self.properties[label].get(f'{self.target_property}_mean', None)

        return image, label_index, target_score

# image
class ImageDataPlus(Dataset):
    """
    path: Path to the directory where the images are saved. 
    transform: A torchvision.transforms object that will be applied to the images
    """

    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.images = []
        self.labels = []
        self.categories = []

        # Look through each sub-directory in the path
        for images in os.listdir(self.path):
            if images[-4:] == '.jpg':
                self.labels.append(images[:-4])
                self.images.append(os.path.join(self.path, images))
                self.categories.append(images[:-4])

        self.categories = sorted(self.categories)
        self.label_to_index = {self.categories[i]: i for i in range(len(self.categories))}

    def __len__(self):
        # Return the total number of images
        return len(self.images)

    def __getitem__(self, idx):
        # Open and send one image and its label
        image_path = self.images[idx]
        image =  Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        label = self.label_to_index[self.labels[idx]]
        return image, label
    

# dreamsim
class TwoAFCDataset_nights(Dataset):
    def __init__(self, root_dir: str, split: str = "train", load_size: int = 224,
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
                 preprocess: str = "DEFAULT", **kwargs):
        self.root_dir = root_dir
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[self.csv['votes'] >= 6] # Filter out triplets with less than 6 unanimous votes
        self.split = split
        self.load_size = load_size
        self.interpolation = interpolation
        self.preprocess_fn = get_preprocess_fn(preprocess, self.load_size, self.interpolation)

        if self.split == "train" or self.split == "val":
            self.csv = self.csv[self.csv["split"] == split]
        elif split == 'test_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == True]
        elif split == 'test_no_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == False]
        else:
            raise ValueError(f'Invalid split: {split}')

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        id = self.csv.iloc[idx, 0]
        p = self.csv.iloc[idx, 2].astype(np.float32)
        img_ref = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 4])))
        img_left = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 5])))
        img_right = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 6])))
        return img_ref, img_left, img_right, p, id
    
# perceptual similarity
class TwoAFCDataset_bapps(BaseDataset):
    def initialize(self, dataroots, load_size=64):
        if(not isinstance(dataroots,list)):
            dataroots = [dataroots,]
        self.roots = dataroots
        self.load_size = load_size

        # image directory
        self.dir_ref = [os.path.join(root, 'ref') for root in self.roots]
        self.ref_paths = make_dataset(self.dir_ref)
        self.ref_paths = sorted(self.ref_paths)

        self.dir_p0 = [os.path.join(root, 'p0') for root in self.roots]
        self.p0_paths = make_dataset(self.dir_p0)
        self.p0_paths = sorted(self.p0_paths)

        self.dir_p1 = [os.path.join(root, 'p1') for root in self.roots]
        self.p1_paths = make_dataset(self.dir_p1)
        self.p1_paths = sorted(self.p1_paths)

        transform_list = []
        transform_list.append(transforms.Resize(load_size))
        transform_list += [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # judgement directory
        self.dir_J = [os.path.join(root, 'judge') for root in self.roots]
        self.judge_paths = make_dataset(self.dir_J,mode='np')
        self.judge_paths = sorted(self.judge_paths)

    def __getitem__(self, index):
        p0_path = self.p0_paths[index]
        p0_img_ = Image.open(p0_path).convert('RGB')
        p0_img = self.transform(p0_img_)

        p1_path = self.p1_paths[index]
        p1_img_ = Image.open(p1_path).convert('RGB')
        p1_img = self.transform(p1_img_)

        ref_path = self.ref_paths[index]
        ref_img_ = Image.open(ref_path).convert('RGB')
        ref_img = self.transform(ref_img_)

        judge_path = self.judge_paths[index]
        # judge_img = (np.load(judge_path)*2.-1.).reshape((1,1,1,)) # [-1,1]
        judge_img = np.load(judge_path).reshape((1,1,1,)) # [0,1]

        judge_img = torch.FloatTensor(judge_img)

        return {'p0': p0_img, 'p1': p1_img, 'ref': ref_img, 'judge': judge_img,
            'p0_path': p0_path, 'p1_path': p1_path, 'ref_path': ref_path, 'judge_path': judge_path}

class ImageMemory(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, subset=None):
        """
        Args:
            csv_path (string): Path to the csv file with memorability scores.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            subset (string, optional): Can be 'train', 'val', or None.
        """
        all_data = pd.read_csv(csv_path)
        all_data = all_data.dropna(subset=['cr'])
        
        split_idx = int(0.8 * len(all_data))

        if subset == "train":
            self.data = all_data[:split_idx]
        elif subset == "val":
            self.data = all_data[split_idx:]
        else:
            self.data = all_data

        self.root_dir = root_dir
        self.transform = transform
        self.targets = self.data['cr'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['file_path'])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        memorability_score = self.data.iloc[idx]['cr']
        return image, memorability_score
    
    def get_by_category(self, category, image_number=None):
        filtered_data = self.data[self.data['file_path'].str.contains(f"/{category}/")]
        if image_number is not None:
            filtered_data = filtered_data.iloc[[image_number - 1]]

        images_scores = []
        for _, row in filtered_data.iterrows():
            img_path = os.path.join(self.root_dir, row['file_path'])
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            memorability_score = row['cr']
            images_scores.append((image, memorability_score))
        
        return images_scores
    
# psiz
class PsizData(torch.utils.data.Dataset):

    def __init__(self, directory, data_path, type, transform=None):
        super().__init__()
        self.directory = directory
        self.data_path = data_path
        self.transform = transform
        self.type = type
        self.rank8_data, self.rank2_data = self.get_all_data(self.directory)
        self.rank8_trial = self.process_data(self.rank8_data, 8)
        self.rank2_trial = self.process_data(self.rank2_data, 2)
        
    @staticmethod
    def get_json_files(directory):
        return [f for f in os.listdir(directory) if f.endswith('.json')]

    @staticmethod
    def get_data(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return [data['data'][0]['subunits']]

    def get_all_data(self, directory):
        json_files = self.get_json_files(directory)
        rank8_data, rank2_data = [], []
        for json_file in json_files:
            single_data = self.get_data(os.path.join(directory, json_file))
            if single_data[0][0]['kind'] == 'rank:8rank2':
                rank8_data.extend(single_data)
            elif single_data[0][0]['kind'] == 'rank:2rank1':
                rank2_data.extend(single_data)
        return rank8_data, rank2_data

    def process_data(self, data, len):
        trials = []
        for d in data:
            for sub in d:
                interactions = sub['interactions']
                trial = []
                for i, interaction in enumerate(interactions):
                    detail = interaction['detail']
                    if detail.startswith("ilsvrc_2012_val"):
                        detail = "/".join(detail.split("/")[:1] + detail.split("/")[2:])  # remove the second part of the path
                    image_path = os.path.join(self.data_path, detail)
                    image = Image.open(image_path).convert("RGB")

                    if self.transform:
                        image = self.transform(image)
                    else:
                        image = transforms.ToTensor()(image)
                    trial.append(image)
                    if i == len + 1:
                        detail = interaction['detail']
                        for j in range(len):
                            if detail == interactions[j]['detail']:
                                trial[1], trial[j] = trial[j], trial[1]  
                    if i == len + 2:
                        detail = interaction['detail']
                        for j in range(len):
                            if detail == interactions[j]['detail']:
                                trial[2], trial[j] = trial[j], trial[2]   
                        break    
                trials.append(torch.stack(trial))
        return torch.stack(trials)

    def __len__(self):
        return len(self.rank8_trial) + len(self.rank2_trial)

    def __getitem__(self, idx):
        if self.type == 'rank8':
            return self.rank8_trial[idx]
        else:
            return self.rank2_trial[idx]


if __name__ == "__main__":
    import pandas as pd
    path = '../data/multi'
    stimulus_category_labels = pd.read_csv(os.path.join(path, 'stimuli/stimulus_category_labels.txt'))
    order = stimulus_category_labels['alphorder'].to_numpy()
    order -= 1
    label_concept = torch.load(os.path.join('../data', 'variables/label_multi_concept.pt'))
    directory1 = os.path.join(path, 'stimuli/set1')
    directory2 = os.path.join(path, 'stimuli/set2')
    dataset1 = MultiDataset(directory1, order)
    dataset2 = MultiDataset(directory2, order)