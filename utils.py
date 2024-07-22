import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid


def plot_bar(names, values, num_display=7, figsize=(1, 2)):
    # names: list of names
    # values: tensor of values
    # num_display: number of names to display
    # select top num_display values
    values, indices = torch.topk(values, num_display)
    names = [names[i] for i in indices]
    # plot bar using seaborn
    plt.figure(figsize=figsize)

    # exchange x and y, color changed with values
    sns.barplot(
        x=values.detach().cpu().numpy(), y=names, hue=names,
        palette="Blues_d", legend=False,
    )
    # remove x axis and ticks
    plt.xlabel('')
    # remove ticks but keep labels in x axis
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    
    # remove y axis
    plt.ylabel('')
    # set yticks by names
    plt.yticks(range(len(names)), names)
    # remove ticks but keep labels in y axis
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)

    # remove border
    sns.despine(left=True, bottom=True)

    quantile_90 = 0.3632
    plt.axvline(x=quantile_90, ymax=1, color='black', linestyle='--', linewidth=1)
    plt.show()
    # save figure
    # plt.savefig('barplot_concept_embeddings.svg', format='svg')

def plot_indices_grid(images, shape=(2,3), size=(224, 224)):

    # Transform to resize images and convert them to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    images = [transform(image) for image in images]

    # Convert list of images to a single tensor
    image_grid = torch.stack(images)

    # Use make_grid to create a grid layout
    grid = make_grid(image_grid, nrow=shape[1])

    # Plot the grid
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')

    plt.show()

if __name__ == "__main__":

    # plot_bar
    path_data = '../data'
    label_concept = torch.load(os.path.join(path_data, 'variables/label_dim.pt'))
    c_embeddings = torch.randn(42) ** 2
    plot_bar(label_concept, c_embeddings, 7)