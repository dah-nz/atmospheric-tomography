import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

from phase_screen import get_kolmogorov_screen

batch_size = 32
n_coeffs = 64

n_train = 100
n_test = n_train / 5

class KolmogorovDataset(torch.utils.data.Dataset):
    def __init__(self, dim, D_r0, train=True, i0=0):
        self.dim = dim
        self.D_r0 = D_r0
        self.train = train
        if train:
            self.len = batch_size * n_train
        else:
            self.len = batch_size * n_test
        self.i0 = i0
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        if self.train:
            x = get_kolmogorov_screen(self.dim, self.D_r0, seed=idx)
        else:
            x = get_kolmogorov_screen(self.dim, self.D_r0, seed=(batch_size*n_test)+idx)
        return self.to_torch(x)
    def to_torch(self, x):
        r = np.real(x)
        i = np.imag(x)
        return torch.from_numpy(np.array([r, i], dtype=np.float32))
    def to_numpy(self, x):
        x = x.numpy()        
        return np.array(x[0] + 1j*x[1], dtype=np.complex128)

        

kernel_size = 3
channels_in = 2
k2 = 16
k3 = 32
k4 = 64
last_layer_dim = 31
LRELU_grad = 0.01

class Encoder(nn.Module):
    def __init__(self, n_coeffs):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(channels_in, k2, kernel_size, stride=2, padding=1, bias=False),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Conv2d(k2, k3, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(k3),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Conv2d(k3, k4, kernel_size, stride=2, padding=0, bias=False),
            nn.LeakyReLU(LRELU_grad, True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(last_layer_dim * last_layer_dim * k4, n_coeffs * 2, bias=False),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Linear(n_coeffs * 2, n_coeffs, bias=False),
        )
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n_coeffs):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(n_coeffs, n_coeffs * 2, bias=False),
            nn.LeakyReLU(LRELU_grad, True),
            nn.Linear(n_coeffs * 2, last_layer_dim * last_layer_dim * k4, bias=False),
            nn.LeakyReLU(LRELU_grad, True),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(k4, last_layer_dim, last_layer_dim))
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(k4, k3, kernel_size, stride = 2, output_padding=0, bias=False),
            nn.BatchNorm2d(k3),
            nn.LeakyReLU(LRELU_grad, True),
            nn.ConvTranspose2d(k3, k2, kernel_size, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(k2),
            nn.LeakyReLU(LRELU_grad, True),
            nn.ConvTranspose2d(k2, channels_in, kernel_size, stride=2, padding=0, output_padding=1, bias=False)
        )
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        return x

def main():
    D_r0 = 15
    grid_size = 256

    train_data = KolmogorovDataset(grid_size, D_r0, i0=0)
    test_data = KolmogorovDataset(grid_size, D_r0, i0=(batch_size*n_train)+1)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    loss_fn = torch.nn.MSELoss()
    lr = 0.001

    torch.manual_seed(0)

    encoder = Encoder(n_coeffs)
    decoder = Decoder(n_coeffs)

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()},
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Selected device: {}".format(device))

    encoder.to(device)
    decoder.to(device)

    num_epochs = 50
    diz_loss = {"train_loss":[], 'val_loss':[]}

    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
        val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
        print("EPOCH {}/{}\ttrain loss {}\tval loss {}".format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        if (epoch + 1) % 10 == 0:
            plot_ae_outputs(encoder, decoder, test_data, device, epoch, n=10)

    torch.save(encoder, 'data/encoder.torch')
    torch.save(decoder, 'data/decoder.torch')

    plot_losses(diz_loss)
    plot_generated_images(encoder, decoder, test_loader, device, test_data)
    plt.show()

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []

    for image_batch in dataloader:
        image_batch = image_batch.to(device)

        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)

        loss = loss_fn(decoded_data, image_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("\tPartial train loss (single batch): {}".format(loss.data))
        train_loss.append((loss.detach().cpu().numpy()))
    return np.mean(train_loss)

def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        conc_out = []
        conc_label = []

        for image_batch in dataloader:
            image_batch = image_batch.to(device)

            encoded_data = encoder(image_batch)
            decoded_data = decoder(encoded_data)

            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())

        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        val_loss = loss_fn(conc_out, conc_label)

    return val_loss.data

def plot_ae_outputs(encoder, decoder, dataset, device, epoch, n=10):
    plt.figure(figsize=(16,4.5))
    plt.title("Epoch {}".format(epoch))
    encoder.eval()
    decoder.eval()
    for i in range(n):
        with torch.no_grad():
            img = dataset[i].unsqueeze(0).to(device)
            rec_img = decoder(encoder(img)).cpu().squeeze()
            rec_img = np.abs(dataset.to_numpy(rec_img))
        img = img.cpu().squeeze()
        img = np.abs(dataset.to_numpy(img))

        ax = plt.subplot(2, n, i+1)
        plt.imshow(img, cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title("Original images")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img, cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n / 2:
            ax.set_title('Reconstructed images')
    # plt.show()

def plot_losses(loss_data):
    plt.figure(figsize=(10,8))
    plt.semilogy(loss_data['train_loss'], label='Train')
    plt.semilogy(loss_data['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    # plt.show()

def plot_generated_images(encoder, decoder, test_loader, device, ds=None):
    with torch.no_grad():
        images = next(iter(test_loader))
        images = images.to(device)
        latent = encoder(images)
        latent = latent.cpu()

        mean = latent.mean(dim=0)
        std = (latent - mean).pow(2).mean(dim=0).sqrt()
        latent = torch.randn(batch_size, n_coeffs) * std + mean
        latent = latent.to(device)
        img_recon = decoder(latent)
        img_recon = img_recon.cpu()
        img_recon = torch.tensor(np.array([np.abs(ds.to_numpy(x)) for x in img_recon]))
        img_recon = img_recon.unsqueeze(1)
        
        fig, ax = plt.subplots(figsize=(20, 8.5))
        import torchvision
        img = torchvision.utils.make_grid(img_recon, 8, 5)
        img = img.numpy()
        img = img[0, :, :]
        plt.imshow(img, cmap='gist_gray')

        latent = torch.diag(torch.ones(n_coeffs)) * mean
        latent = latent.to(device)
        img_recon = decoder(latent)
        img_recon = img_recon.cpu()
        img_recon = torch.tensor(np.array([np.abs(ds.to_numpy(x)) for x in img_recon]))
        img_recon = img_recon.unsqueeze(1)
        
        fig, ax = plt.subplots(figsize=(20, 8.5))
        img = torchvision.utils.make_grid(img_recon, 8, 5)
        img = img.numpy()
        img = img[0, :, :]
        # img /=  np.amax(img) if np.amax(img) > -np.amin(img) else -np.amin(img)
        # img = np.transpose(img, (1, 2, 0))
        plt.imshow(img, cmap='gist_gray')
    # plt.show()

if __name__ == "__main__":
    main()