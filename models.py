import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = points.transpose(2, 1)           
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x)))  
        x = F.relu(self.bn3(self.conv3(x)))  
        x = torch.max(x, 2)[0]  

        x = F.relu(self.bn4(self.fc1(x))) 
        x = F.relu(self.bn5(self.dropout(self.fc2(x)))) 
        x = self.fc3(x)  
        out = F.log_softmax(x, dim=1)
        
        return out



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, num_seg_classes, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        x = points.transpose(2, 1)  
        x1 = F.relu(self.bn1(self.conv1(x)))     
        x2 = F.relu(self.bn2(self.conv2(x1)))    
        x3 = F.relu(self.bn3(self.conv3(x2)))    
       
        global_feat = torch.max(x3, 2, keepdim=True)[0]  
        global_feat = global_feat.repeat(1, 1, x.size(2))  

        concat_feat = torch.cat([x1, global_feat], 1)  
        
        x = F.relu(self.bn4(self.conv4(concat_feat)))  
        x = F.relu(self.bn5(self.dropout(self.conv5(x))))  
        x = self.conv6(x)  
        out = x.transpose(2, 1).contiguous()
        
        return out


# PointNet++ Implementation

class PointNetPPClassification(nn.Module):
    def __init__(self, num_classes=3):
        super(PointNetPPClassification, self).__init__()

        # Set abstraction layers (3 levels, like the original PointNet++)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=0, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024])  

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: (B, N, 3) - input point cloud coordinates
        """
        B, N, _ = x.shape

        l1_xyz, l1_points = self.sa1(x, None)          # → (B, 512, 128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # → (B, 128, 256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # → (B, 1, 1024)

        x = l3_points.squeeze(1)  # shape: (B, 1024)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)  # classification output


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        """
        npoint: number of sampled points
        radius: radius for local neighborhood
        nsample: number of neighbors per sampled point
        in_channel: input feature channel size (e.g., 0 or 3 for just XYZ)
        mlp: list of output sizes for MLP layers, e.g., [64, 64, 128]
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # local XYZ will be concatenated

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        xyz: (B, N, 3) - input XYZ coordinates
        points: (B, N, C) - input point features, or None
        Return:
            new_xyz: (B, npoint, 3)
            new_points: (B, npoint, mlp[-1])
        """
        B, N, _ = xyz.shape

        # 1. Sample keypoints
        idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, idx)  # (B, npoint, 3)

        # 2. Group neighboring points
        group_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)
        grouped_xyz = index_points(xyz, group_idx)  # (B, npoint, nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)  # local coordinates

        if points is not None:
            grouped_points = index_points(points, group_idx)  # (B, npoint, nsample, C)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B, npoint, nsample, C+3)
        else:
            grouped_points = grouped_xyz  # (B, npoint, nsample, 3)

        # 3. Apply mini-PointNet
        grouped_points = grouped_points.permute(0, 3, 2, 1)  # (B, C+3, nsample, npoint)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))

        new_points = torch.max(grouped_points, 2)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])

        return new_xyz, new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: point cloud data, (B, N, 3)
        npoint: number of points to sample
    Return:
        centroids: sampled point indices, (B, npoint)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: search radius
        nsample: max number of points in the neighborhood
        xyz: all points, (B, N, 3)
        new_xyz: query points (e.g., sampled centers), (B, S, 3)
    Return:
        group_idx: indices of grouped points, (B, S, nsample)
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    
    group_idx = torch.arange(N, dtype=torch.long).to(xyz.device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, -1)
    
    group_idx[sqrdists > radius ** 2] = N  # mark far points with N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # keep closest nsample
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]  # replace invalid with first point
    return group_idx

def index_points(points, idx):
    """
    Input:
        points: input points data, (B, N, C)
        idx: sample index data, (B, S) or (B, S, nsample)
    Return:
        new_points: indexed points data
    """
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long).to(points.device).view(B, 1, 1)

    if idx.dim() == 2:
        return points[batch_indices, idx, :]
    elif idx.dim() == 3:
        batch_indices = batch_indices.expand(-1, idx.shape[1], idx.shape[2])
        return points[batch_indices, idx, :]
