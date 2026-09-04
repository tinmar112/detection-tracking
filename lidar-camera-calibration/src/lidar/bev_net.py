from torch import nn

from lidar.bev_backbone import BEVBackbone  #type: ignore
from lidar.pfn import PFNLayer  #type: ignore
from lidar.pillarization import Pillarization  #type: ignore


class PointPillars(nn.Module):

    def __init__(self):
        super().__init__()

        self.pfn = PFNLayer(in_channels=9, out_channels=64)

        self.backbone = BEVBackbone(in_channels=64)

        self.cls_head = nn.Conv2d(128, 1, 1) # Probability for car

        self.reg_head = nn.Conv2d(128, 6, 1) # x,y,z,w,l,h of car

        self.yaw_head = nn.Conv2d(128, 2, 1) # car orientation

    def forward(self, pillars, coords,
                pillarization: Pillarization):

        pillar_features = self.pfn(pillars)

        bev = pillarization.scatter_bev(pillar_features, coords)
        bev = bev.unsqueeze(0) # add a batch size

        features = self.backbone(bev)

        cls = self.cls_head(features)
        reg = self.reg_head(features)
        yaw = self.yaw_head(features)

        return cls, reg, yaw
