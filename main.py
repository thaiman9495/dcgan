import torch
import torchvision
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_root_folder = "data/celeba"
    img_size = 64
    batch_size = 128

    # Define some transformations to be applied to the images of the dataset
    list_transformation = [torchvision.transforms.Resize(img_size),
                           torchvision.transforms.CenterCrop(img_size),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    data_transoform = torchvision.transforms.Compose(list_transformation)

    # Create a torch dataset
    dataset = torchvision.datasets.ImageFolder(root=data_root_folder, transform=data_transoform)
    
    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    

    
