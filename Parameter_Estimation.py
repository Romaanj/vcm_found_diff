import torch
import torch.nn as nn


# Parameter Estimation Network
class ParameterEstimationNetwork(nn.Module):
    def __init__(self, input_channels=5, num_params=9):
        super(ParameterEstimationNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1), # B X 64 X H/2 X W/2
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # B X 128 X H/4 X W/4
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # B X 256 X H/8 X W/8
            nn.SiLU(),
            nn.Conv2d(256, num_params, kernel_size=3, stride=1, padding=1), # B X 9 X H' X W'
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling # B X 9 X 1 X 1
        )
    
    def forward(self, y, lambda_param):
    # y shape: (B, C, H, W)
      B, C, H, W = y.shape

    # lambda_param: float or shape=(1,)  -> scalar
    # 1) lambda를 (B,1,1,1) 형태로 바꾼 뒤
      lambda_expanded = lambda_param.view(B,1,1,1).expand(-1, -1, H, W)
      lambda_expanded = lambda_param.view(1, 1, 1, 1)  # shape (1,1,1,1)
    # 2) batch, H, W에 맞게 확장
      lambda_expanded = lambda_expanded.expand(B, 1, H, W)  # shape (B,1,H,W)

    # 3) 채널 차원에서 concat
    # => input_tensor shape: (B, C+1, H, W)
      input_tensor = torch.cat((y, lambda_expanded), dim=1)

      params = self.net(input_tensor).view(B, -1) # B X 9
      gamma = params[:, :-1] # B X 8
      t = torch.sigmoid(params[:, -1]) # B X 1
      return gamma, t