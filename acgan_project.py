import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
from pytorch_image_generation_metrics import ImageDataset
from torch.utils.data import DataLoader
from pytorch_image_generation_metrics import get_inception_score
from pytorch_msssim import ms_ssim
import pandas as pd
import torch

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument('--adversarial_loss', type=str, default='bce', choices=['bce', 'cross_entropy', 'mse'], help='The loss function to use during training')
parser.add_argument('--auxiliary_loss', type=str, default='cross_entropy', choices=['bce', 'cross_entropy', 'mse'], help='The loss function to use during training')
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FAMNIST','CIFAR10', 'CIFAR100'], help='The dataset to use')
parser.add_argument("--output_dir", type=str, default="output", help="directory to save images and plots")  # New argument
opt = parser.parse_args()
print(opt)

# Create necessary directories
os.makedirs(os.path.join(opt.output_dir, "images"), exist_ok=True)

cuda = True if torch.cuda.is_available() else False

def get_loss_function(name):
    if name == 'bce':
        return torch.nn.BCELoss()
    elif name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif name == 'mse':
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")

def get_dataset(name):
    if name == 'MNIST':
        os.makedirs("../../data/MNIST", exist_ok=True)
        dataset = datasets.MNIST(
            "../../data/MNIST",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        )
        return dataset
    elif name == 'FAMNIST':
        os.makedirs("../../data/FAMNIST", exist_ok=True)
        dataset = datasets.FAMNIST(
            "../../data/FAMNIST",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        )
        return dataset
    elif name == 'CIFAR10':
        os.makedirs("../../data/CIFAR10", exist_ok=True)
        dataset = datasets.CIFAR10(
            "../../data/CIFAR10",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        )
        return dataset
    elif name == 'CIFAR100':
        os.makedirs("../../data/CIFAR100", exist_ok=True)
        dataset = datasets.CIFAR100(
            "../../data/CIFAR100",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        )
        return dataset
    else:
        raise ValueError(f"Unknown loss function: {name}")

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

# Loss functions
adversarial_loss = get_loss_function(opt.adversarial_loss)
auxiliary_loss = get_loss_function(opt.auxiliary_loss)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader

dataset = get_dataset(opt.dataset)

# Split dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done, epoch):
    """Saves individual generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)

    # Ensure the output directory exists
    os.makedirs(os.path.join(opt.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, f"images/{epoch}"), exist_ok=True)

    # Save each image individually
    for i in range(gen_imgs.size(0)):
        save_image(gen_imgs.data[i], os.path.join(opt.output_dir, f"images/{epoch}/{batches_done}_{i}.png"), normalize=True)


# Store losses and accuracies
loader = {
  'train' : train_loader,
  'test' : test_loader
  }

g_losses_test = []
d_losses_test = []
d_accuracies_test = []

g_losses_train = []
d_losses_train = []
d_accuracies_train = []

IS_score = []
IS_std_score = []
ms_ssim_scores = []
# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    for mode in ['train','test']:
        generator.train() if mode == 'train' else generator.eval()
        discriminator.train() if mode == 'train' else discriminator.eval()
        for i, (imgs, labels) in enumerate(loader[mode]):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            if mode == 'train' : optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

            if mode == 'train' : g_loss.backward()
            if mode == 'train' : optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if mode == 'train' : optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = discriminator(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            if mode == 'train' : d_loss.backward()
            if mode == 'train' : optimizer_D.step()

            if mode == 'train' :
               g_losses_train.append(g_loss.item())
               d_losses_train.append(d_loss.item())
               d_accuracies_train.append(d_acc)
            if mode == 'test' :
               g_losses_test.append(g_loss.item())
               d_losses_test.append(d_loss.item())
               d_accuracies_test.append(d_acc)

            print(
                "[Mode %s]  [Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                % (mode, epoch, opt.n_epochs, i, len(loader[mode]), d_loss.item(), 100 * d_acc, g_loss.item())
            )
            batches_done = epoch * len(loader[mode]) + i
            if batches_done % opt.sample_interval == 0 and mode == 'train':
                sample_image(n_row=10, batches_done=batches_done, epoch=epoch)
                samples = ImageDataset(os.path.join(opt.output_dir, f"images/{epoch}/"), exts=['png', 'jpg'])
                samples_loader = DataLoader(samples, batch_size=50, num_workers=4)
                # Inception Score
                IS, IS_std = get_inception_score(samples_loader)
                IS_score.append(IS)
                IS_std_score.append(IS_std)
                # Calculate MS-SSIM for generated images
                transform = transforms.Resize((256, 256))
                # Apply the transform
                ms_ssim_score = ms_ssim(transform(real_imgs), transform(gen_imgs), data_range=1.0, size_average=True).item()
                ms_ssim_scores.append(ms_ssim_score)


# Save scores to CSV files
train_scores = {
    "g_losses_train": g_losses_train,
    "d_losses_train": d_losses_train,
    "d_accuracies_train": d_accuracies_train
}
test_scores = {
    "g_losses_test": g_losses_test,
    "d_losses_test": d_losses_test,
    "d_accuracies_test": d_accuracies_test
}
metrics = {
    "IS_score": IS_score,
    "IS_std_score": IS_std_score,
    "ms_ssim_scores": ms_ssim_scores
}
pd.DataFrame(train_scores).to_csv(os.path.join(opt.output_dir, "train_scores.csv"))
pd.DataFrame(test_scores).to_csv(os.path.join(opt.output_dir, "test_scores.csv"))
pd.DataFrame(metrics).to_csv(os.path.join(opt.output_dir, "metrics_scores.csv"))