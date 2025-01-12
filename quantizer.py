import torch.nn as nn
class UniformQuantizer(nn.Module):
    def __init__(self, num_bits=4, alpha = 0.9):
        super().__init__()
        self.num_bits = num_bits
        self.alpha = alpha

    # 일반적인 uniform quantization
    def forward(self, x):
        # 분포 분석에서 본 것처럼 대부분의 값이 [-3, 3] 범위 내에 있습니다
        # 약간의 여유를 두어 [-4, 4] 범위로 설정합니다
        min_val, max_val = -4.0, 4.0

        # 양자화 스텝 크기를 계산합니다
        # 예: 8비트의 경우 256개의 레벨로 나눕니다
        num_levels = 2 ** self.num_bits
        step_size = (max_val - min_val) / (num_levels - 1)

        # 값을 양자화 레벨로 변환합니다
        x_normalized = (x - min_val) / step_size
        x_quantized = torch.round(x_normalized)

        # 범위를 벗어나는 값들을 클리핑합니다
        x_quantized = torch.clamp(x_quantized, min = torch.tensor(0,device=x_quantized.device), max = torch.tensor(num_levels - 1, device=x_quantized.device))

        # 다시 원래 값의 범위로 변환합니다
        x_dequantized = x_quantized * step_size + min_val

        # 압축 비율 계산 추가
        compression_ratio, bpp = self.calculate_compression_ratio(x, x_quantized)

        # 계산된 정보를 로깅하거나 저장
        self.last_compression_stats = {
            'compression_ratio': compression_ratio,
            'bpp': bpp
        }

        # 학습을 위해 Straight-Through Estimator를 사용합니다
        #if self.training:
            # forward: 양자화된 값을 사용
            # backward: 양자화되지 않은 원본 기울기를 사용
            #x_dequantized = x + (x_dequantized - x).detach()

        return x_dequantized
    def forward_diffusion_style(self, x, alpha_bar_t):
      """
      1) Diffusion forward-style 노이즈를 주입한 뒤
      2) Uniform 양자화
      """
      eps = torch.randn_like(x)
      x_noisy = torch.sqrt(alpha_bar_t) * x + \
                  torch.sqrt(1.0 - alpha_bar_t) * eps

      x_dequantized = self.forward(x_noisy)
      return x_dequantized

    # Quantization Error가 Gaussian이 나오도록 하는 Quantization
    def forward_with_gaussian_error(self, x):
        min_val, max_val = -4.0, 4.0
        num_levels = 2 ** self.num_bits
        step_size = (max_val - min_val) / (num_levels - 1)

        # 가능한 양자화 레벨들을 모두 생성
        levels = torch.arange(num_levels, device=x.device) * step_size + min_val

        # 각 입력값에 대해 Gaussian 분포를 따르는 레벨 선택
        x_flat = x.flatten()
        quantized = torch.zeros_like(x_flat)

        for i, val in enumerate(x_flat):
        # 각 레벨과 입력값 사이의 거리 계산
          distances = ((levels - val) / step_size)**2

        # 수치적 안정성을 위해 거리값에서 최솟값을 빼줍니다
          distances = distances - distances.min()

        # 확률 계산 시 적절한 scale factor 사용
          sigma = 1.0  # 이 값을 조정하여 분포의 퍼짐 정도를 제어할 수 있습니다
          probs = torch.exp(-0.5 * distances**2 / (sigma**2))

        # 확률이 너무 작은 값들은 제외 (수치적 안정성을 위해)
          mask = probs > 1e-10
          if mask.sum() == 0:  # 모든 확률이 너무 작은 경우
            # 가장 가까운 레벨 선택
              level_idx = torch.argmin(distances)
          else:
            # 유효한 확률값들만 사용
            valid_probs = probs[mask]
            valid_levels = levels[mask]

            # 확률 정규화
            valid_probs = valid_probs / valid_probs.sum()

            # 레벨 선택
            selected_idx = torch.multinomial(valid_probs, 1)
            quantized[i] = valid_levels[selected_idx]

        gaussian_quantized_latent=quantized.reshape(x.shape)

        compression_ratio, bpp = self.calculate_compression_ratio(x, gaussian_quantized_latent)

            # 계산된 정보를 로깅하거나 저장
        self.last_gaussian_compression_stats = {
                'compression_ratio': compression_ratio,
                'bpp': bpp
            }
        return gaussian_quantized_latent


    def get_quantization_error(self, x):
        # 양자화 오차를 계산하여 반환합니다
        with torch.no_grad():
            x_quantized = self.forward(x)
            error = x_quantized - x
        return error

    def get_gaussian_quantization_error(self, x):

      with torch.no_grad():
        x_quantized = self.forward_with_gaussian_error(x)
        error = x_quantized - x
      return error