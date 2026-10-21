import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import zipfile

# Set random seed for reproducibility
manualSeed = 42
torch.manual_seed(manualSeed)

# ==========================================
# 1. Configuration and Data Loading
# ==========================================
# Path to the extracted Food-101 images directory
image_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset",
    "food-101",
    "food-101",
    "images",
)

# Hyperparameters
batch_size = 128
image_size = 64 # Resize to 64x64 for performance
nc = 3          # Number of channels (RGB)
nz = 100        # Size of z latent vector (i.e. size of generator input)
ngf = 64        # Size of feature maps in generator
ndf = 64        # Size of feature maps in discriminator
num_epochs = 5  # Start with 5 epochs for practical assignment; increase for better results
lr = 0.0002
beta1 = 0.5     # Beta1 hyperparam for Adam optimizers

# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

try:
    dataset = torchvision.datasets.ImageFolder(root=image_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
except Exception as e:
    print(f"Error loading data from {image_dir}: {e}")
    print("Please ensure the dataset is extracted and has class subfolders.")
    dataset = None


# ==========================================
# 2. Vanilla GAN Implementation
# ==========================================

class VanillaGenerator(nn.Module):
    def __init__(self):
        super(VanillaGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, nc * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        output = self.main(input)
        return output.view(-1, nc, image_size, image_size)

class VanillaDiscriminator(nn.Module):
    def __init__(self):
        super(VanillaDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nc * image_size * image_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.main(input)


# ==========================================
# 3. DCGAN Implementation
# ==========================================

# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGANGenerator(nn.Module):
    def __init__(self):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# ==========================================
# 4. Training Function
# ==========================================

def train_gan(generator, discriminator, dataloader, model_name="GAN"):
    print(f"Starting Training for {model_name}...")
    
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device) if model_name == "DCGAN" else torch.randn(64, nz, device=device)
    
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    img_list = []
    G_losses = []
    D_losses = []
    
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            if model_name == "DCGAN":
                noise = torch.randn(b_size, nz, 1, 1, device=device)
            else:
                noise = torch.randn(b_size, nz, device=device)
                
            fake = generator(noise)
            label.fill_(fake_label)
            
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            optimizerG.step()
            
            # Output training stats
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        
    return G_losses, D_losses, img_list

# ==========================================
# 5. Execution Pipeline
# ==========================================
if __name__ == '__main__':
    if dataset is not None:
        # Initialize Models
        print("Initializing Vanilla GAN...")
        vanilla_netG = VanillaGenerator().to(device)
        vanilla_netD = VanillaDiscriminator().to(device)

        print("Initializing DCGAN...")
        dcgan_netG = DCGANGenerator().to(device)
        dcgan_netG.apply(weights_init)
        dcgan_netD = DCGANDiscriminator().to(device)
        dcgan_netD.apply(weights_init)

        # Train Vanilla GAN
        vanilla_G_losses, vanilla_D_losses, vanilla_img_list = train_gan(vanilla_netG, vanilla_netD, dataloader, "Vanilla GAN")

        # Train DCGAN
        dcgan_G_losses, dcgan_D_losses, dcgan_img_list = train_gan(dcgan_netG, dcgan_netD, dataloader, "DCGAN")
        
        print("Training complete! Models initialized successfully!")
        print("Note: Training GANs on CPU may be slow. Consider adjusting num_epochs or batch_size.")
