import torch
import torch.nn as nn
import torch.utils.data


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size, 4, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.bn1(self.conv1(x)))
        return x


class ResUp(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=4):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        return x


class Encoder(nn.Module):

    def __init__(self, channels, ch=64, latent_channels=512):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, ch, 7, 2, 3)
        self.res_down_block1 = ResDown(ch, 2 * ch)
        self.res_down_block2 = ResDown(2 * ch, 4 * ch)
        self.conv_mu = nn.Conv2d(4 * ch, latent_channels, 4)
        # self.conv_log_var = nn.Conv2d(16 * ch, latent_channels, 4, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_in(x))
        x = self.res_down_block1(x)  # 32
        x = self.res_down_block2(x)  # 8
        z = self.conv_mu(x)  # 1

        return z


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, latent_channels=512):
        super(Decoder, self).__init__()
        self.conv_t_up = nn.ConvTranspose2d(latent_channels, ch * 16, 4, 1)
        self.res_up_block1 = ResUp(ch * 16, ch * 8)
        self.res_up_block2 = ResUp(ch * 8, ch * 4)
        self.res_up_block2 = ResUp(ch * 8, ch * 4)
        self.res_up_block3 = ResUp(ch * 4, ch, scale_factor=2)
        self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.act_fnc(self.conv_t_up(x))  # 4
        x = self.res_up_block1(x)  # 8
        x = self.res_up_block2(x)  # 16
        x = self.res_up_block3(x)  # 32
        x = torch.tanh(self.conv_out(x))

        return x


class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """

    def __init__(self, channel_in=3, ch=64, latent_channels=512):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """

        self.encoder = Encoder(channel_in, ch=ch, latent_channels=latent_channels)
        self.decoder = Decoder(channel_in, ch=ch, latent_channels=latent_channels)

    def forward(self, x):
        encoding = self.encoder(x)
        recon_img = self.decoder(encoding)
        return recon_img


if __name__ == '__main__':

    net = VAE(latent_channels=32)
    x_sample = torch.normal(0,1, (10, 3, 128, 128))
    y_sample = net(x_sample)
    s = 1