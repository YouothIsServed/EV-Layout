import  torch

checkpoint = torch.load('./outputs/checkpoint_1map_main/stage_4/latest_swin_hawp_61.pth')['model']
c = list(checkpoint.keys())
checkpoint_old = torch.load('./outputs/checkpoint/latest_swin_hawp_10.pth')['model']
c2 = list(checkpoint_old.keys())

intersection = list(set(c) & set(c2))

# 差集
difference = list(set(c2) - set(c))