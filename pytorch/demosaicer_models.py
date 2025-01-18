import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F

# Demosaicer model for the raw sensor output shrank to 1D
class Demos(nn.Module):
    def __init__(self):
        super(Demos, self).__init__()
        self.pseudoimage = nn.Conv2d(1, 3, kernel_size=9, stride=1, padding=4, bias=False)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
    def forward(self, input_tensor):
        psd = self.pseudoimage(input_tensor)
        x1 = self.block1(psd)
        c1 = cat((psd, x1), 1)
        x2 = self.block2(c1)
        c2 = cat((x1, x2), 1)
        x3 = self.block3(c2)
        c3 = cat((x2, x3), 1)
        out = self.output(c3)
        return [out, x3, x2, x1, psd]

class Demos3D(nn.Module):
    def __init__(self):
        super(Demos3D, self).__init__()
        self.pseudoimage = nn.Conv2d(3, 3, kernel_size=9, stride=1, padding=4, bias=False)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
    def forward(self, input_tensor):
        psd = self.pseudoimage(input_tensor)
        x1 = self.block1(psd)
        c1 = cat((psd, x1), 1)
        x2 = self.block2(c1)
        c2 = cat((x1, x2), 1)
        x3 = self.block3(c2)
        c3 = cat((x2, x3), 1)
        out = self.output(c3)
        return [out, x3, x2, x1, psd]

#The demosaicing model proposed in Henz
def henz_interpolation_kernel(k):
    intrpl_filter = torch.unsqueeze(torch.arange(1, k, 1), 0)
    intrpl_filter = torch.cat([intrpl_filter, torch.tensor([[k]]), torch.flip(intrpl_filter, dims=[1])], dim=1)
    intrpl_filter = torch.div(intrpl_filter, k)
    intrpl_filter = torch.matmul(torch.transpose(intrpl_filter,0,1), intrpl_filter)
    intrpl_filter = torch.unsqueeze(torch.unsqueeze(intrpl_filter, 0), 0)
    return intrpl_filter

class HenzInput(nn.Module):
    def __init__(self):
        super(HenzInput, self).__init__()
    
    def forward(self, mosaic, submosaic, intrpl_filter):
        submosaic_b = torch.unsqueeze(submosaic[:,0,:,:],1)
        submosaic_g = torch.unsqueeze(submosaic[:,1,:,:],1)
        submosaic_r = torch.unsqueeze(submosaic[:,2,:,:],1)
        
        #interpol_kernel = self.k.to("cuda:0")
        
        intrpl_b = F.conv2d(submosaic_b, intrpl_filter, stride=1, padding=7)
        intrpl_g = F.conv2d(submosaic_g, intrpl_filter, stride=1, padding=7)
        intrpl_r = F.conv2d(submosaic_r, intrpl_filter, stride=1, padding=7)
        
        intrpl = torch.cat([intrpl_b, intrpl_g, intrpl_r], dim=1)
        input_tensor = torch.cat([mosaic, submosaic, intrpl], dim=1)
        return input_tensor

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=(3,3), stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(output_size)
        
    def forward(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        return F.relu(x)
    
class DemosHenz(nn.Module):
    def __init__(self):
        super(DemosHenz, self).__init__()
        self.henz_input = HenzInput()
        self.conv_blk1 = ConvBlock(7, 64)
        self.conv_blk2 = ConvBlock(64, 64)
        self.conv_blk3 = ConvBlock(64, 64)
        self.conv_blk4 = ConvBlock(64, 64)
        self.conv_blk5 = ConvBlock(64, 64)
        self.conv_blk6 = ConvBlock(64, 64)
        self.conv_blk7 = ConvBlock(64, 128)
        self.conv_blk8 = ConvBlock(128, 128)
        self.conv_blk9 = ConvBlock(128, 128)
        self.conv_blk10 = ConvBlock(128, 128)
        self.conv_blk11 = ConvBlock(128, 128)
        self.conv_blk12 = ConvBlock(128, 128)
        self.conv_output = nn.Conv2d(129, 3, kernel_size=(3,3), stride=1, padding=1)
        
    def forward(self, filtered_image, weighted_image, intrpl_filter):
        input_tensor = self.henz_input(filtered_image, weighted_image, intrpl_filter)
        x = self.conv_blk1(input_tensor)
        x = self.conv_blk2(x)
        x = self.conv_blk3(x)
        x = self.conv_blk4(x)
        x = self.conv_blk5(x)
        x = self.conv_blk6(x)
        x = self.conv_blk7(x)
        x = self.conv_blk8(x)
        x = self.conv_blk9(x)
        x = self.conv_blk10(x)
        x = self.conv_blk11(x)
        x = self.conv_blk12(x)
        x = torch.cat([filtered_image, x], dim=1)
        x = self.conv_output(x)
        return F.relu(x)
    
# ISTA-NET adaptation from https://github.com/jianzhangcs/ISTA-Net-PyTorch
class ISTANetBlock(torch.nn.Module):
    def __init__(self):
        super(ISTANetBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv_D = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 33, 33)

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G
        x_pred = x_pred.view(-1, 1089)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]

class ISTANetPlus(torch.nn.Module):
    def __init__(self, num_layers):
        super(ISTANetPlus, self).__init__()
        onelayer = []
        self.num_layers = num_layers

        for i in range(num_layers):
            onelayer.append(ISTANetBlock())

        self.fcs = nn.ModuleList(onelayer)
        # Phi: Sampling matrix
        # Phix: Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
    
# ADMM-NET adaptation from https://github.com/yangyan92/Deep-ADMM-Net