import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)

        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class Pct(nn.Module):
    def __init__(self, args, output_channels=1):
        super(Pct, self).__init__()
        self.args = args
        #self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        # bandgap input dimension as 100
        self.conv1 = nn.Conv1d(203, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        # revised by Rui 20210730 to decrease neuron num 256,265 -> 256,64
        self.gather_local_1 = Local_op(in_channels=256, out_channels=64)

        self.pt_last = Point_Transformer_Last(args)
        # revised by Rui 20210606 TO adjust channel number1280-> 2304 -> 3328 -> 832 --> 1088
        self.conv_fuse = nn.Sequential(nn.Conv1d(1088, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        self.m = nn.Sigmoid()

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=64, radius=0.15, nsample=8, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=32, radius=0.2, nsample=8, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        x = self.m(x)
        x = torch.mul(x, 10)
        # print x; add sigmoid func
        return x

class Point_Transformer_Last(nn.Module):
    # channels change from 256 to 64 due to dimension reduction purpose
    def __init__(self, args, channels=64):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)
        # revised by Rui 20210606 added attention layer
        self.sa5 = SA_Layer(channels)
        self.sa6 = SA_Layer(channels)
        self.sa7 = SA_Layer(channels)
        self.sa8 = SA_Layer(channels)

        self.sa9 = SA_Layer(channels)
        self.sa10 = SA_Layer(channels)
        self.sa11 = SA_Layer(channels)
        self.sa12 = SA_Layer(channels)

        self.sa13 = SA_Layer(channels)
        self.sa14 = SA_Layer(channels)
        self.sa15 = SA_Layer(channels)
        self.sa16 = SA_Layer(channels)

        #self.sa17 = SA_Layer(channels)
        #self.sa18 = SA_Layer(channels)
        #self.sa19 = SA_Layer(channels)
        #self.sa20 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        # revised by Rui 20210606 added attention layer
        x5 = self.sa5(x4)
        x6 = self.sa6(x5)
        x7 = self.sa7(x6)
        x8 = self.sa8(x7)
        x9 = self.sa9(x8)
        x10 = self.sa10(x9)
        x11 = self.sa11(x10)
        x12 = self.sa12(x11)

        x13 = self.sa12(x12)
        x14 = self.sa12(x13)
        x15 = self.sa12(x14)
        x16 = self.sa12(x15)

        #x17 = self.sa12(x16)
        #x18 = self.sa12(x17)
        #x19 = self.sa12(x18)
        #x20 = self.sa12(x19)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
