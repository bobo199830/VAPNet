import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModule(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.norm1 = nn.LayerNorm(opt.in_dim)
        self.norm2 = nn.LayerNorm(opt.in_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.att = nn.MultiheadAttention(opt.in_dim, opt.num_heads, dropout=0.1)

    def forward(self,q,k,v):
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        x1 = self.dropout2(self.att(q,k,v)[0])
        x2 = self.norm1(v+x1)
        return x2.squeeze()


class netVideo(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(512,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,2048),
            nn.BatchNorm1d(2048)
        )
    
    def forward(self,x):
        proj = self.projector(x)
        return F.normalize(proj)


class netAtt(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(384,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,2048),
            nn.BatchNorm1d(2048)
        )
    
    def forward(self,x):
        return F.normalize(self.projector(x))


class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,input_dim)
        self.fc2 = nn.Linear(input_dim,input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.lrelu = nn.ReLU(inplace=True)
        self.linear_means = nn.Linear(input_dim,input_dim)
        self.linear_log_var = nn.Linear(input_dim,input_dim)

    def forward(self,x):
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,input_dim)
        self.fc2 = nn.Linear(input_dim,input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.lrelu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,z):
        x = self.bn1(self.lrelu(self.fc1(z)))
        x = self.lrelu(self.fc2(x))
        return x

class UncertaintyModule(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.encoder = Encoder(opt.in_dim)
        self.decoder = Decoder(opt.in_dim)
    
    def forward(self,q):
        mean, var = self.encoder(q)
        std = torch.exp(0.5*var).cuda()
        eps = torch.randn([q.size(0),q.size(1)]).cuda()
        q = eps*std + mean
        return q, mean, var


class VAPNet(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.vhead = netVideo(opt)
        self.ahead = netAtt(opt)
        self.un = UncertaintyModule(opt)
        self.ca1 = CrossAttentionModule(opt)
        self.ca2 = CrossAttentionModule(opt)
    
    def forward(self,v,cap,cls,des):
        proj_v = self.vhead(v)
        q, mean, var = self.un(cap)
        att = self.ca1(q,cls,cls) + self.ca2(q,des,des)
        proj_a = self.ahead(att)
        return proj_v, proj_a, mean, var