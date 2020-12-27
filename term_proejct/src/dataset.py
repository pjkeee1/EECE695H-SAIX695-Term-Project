import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class CUB(Dataset):
    def __init__(self, root, state=None, data_len=None):
        """
            Note that CUB has 200 classes, but we only use 100 classes in the training step
            Validation is conducted in the remaining 50 classes

            *** Never change the data loader init, len part ***
            *** getitem part can be changed for data augmentation ***
            *** Never include the remaining classes in the training step. It is considered cheating. ***

        """
        if state is None:
            state = ['train', 'val', 'test', 'class_test']
        self.root = root
        self.state = state

        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        
        # 11787개의 img name list load 후 training set과 validation set을 나눔
        img_name_list = []

        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        train_img_name_list = img_name_list[:5804]
        val_img_name_list = img_name_list[5804:8822]
        
        # 11787개의 label name list load 후 training set과 validation set을 나눔
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_label_list = label_list[:5804]
        val_label_list = label_list[5804:8822]

        if self.state == 'train':
            self._imgs = [plt.imread(os.path.join(self.root, 'images', f))
                               for f in train_img_name_list]
            self._labels = [x for x in train_label_list]

        elif self.state == 'val':
            self._imgs = [plt.imread(os.path.join(self.root, 'images', f))
                             for f in val_img_name_list]
            self._labels = [x for x in val_label_list]

        else:
            raise RuntimeError('Invalid state!')

    def __getitem__(self, index):
        """ Data augmentation part

            *** getitem part can be changed for data augmentation ***

        """
        img, target = self._imgs[index], self._labels[index]

        # convert grayscale images into RGB images
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)

        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize((512, 512), Image.BILINEAR)(img)
        img = transforms.CenterCrop((400, 400))(img)
        # img = transforms.Resize((200, 200), Image.BILINEAR)(img)
        img = transforms.Resize((28, 28), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.state == 'train':
            return len(self._train_label)
        elif self.state == 'val':
            return len(self._val_label)


