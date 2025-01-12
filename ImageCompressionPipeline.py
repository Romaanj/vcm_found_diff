import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision.transforms import functional as TF
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import math

from module.quant import NoiseQuant, SteQuant

class ImageCompressionPipeline:
    def __init__(self):
        # Load Diffusion pipeline
        self.diffusion_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", scheduler=DDIMScheduler(), torch_dtype=torch.float16
        ).to("cuda")

        # Extract VAE from pipeline
        self.vae = self.diffusion_pipe.vae
        self.scheduler = self.diffusion_pipe.scheduler

        self.scheduler.set_timesteps(1000)

    def encode_latent(self, image):
        image = image * 2 - 1
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample() * 0.18215
        return latent
    
    # Adaptive Quantization
    def adaptive_quantization(self, y, gamma, mode = 'ste'):
        """
        y: Latent 벡터 (B, C, H, W)
        gamma: 양자화 파라미터 (B, 2 * C) - 채널별 (alpha, beta)
        """

        quant_noise = NoiseQuant()
        quant_ste = SteQuant()
        # gamma에서 alpha와 beta 추출
        channels = 4
        alpha = gamma[:, :channels].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = gamma[:, channels:].unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
    
        # Affine Transformation
        transformed = alpha * y + beta

        # Quantization
        if mode == 'ste':
            quantized = quant_ste(transformed)
        elif mode == 'noise':
            quantized = quant_noise(transformed)
        
        return quantized

    # Inverse Quantization
    def inverse_quantization(self, z_hat, gamma):
        """
        z_hat: Quantized Latent 벡터 (B, C, H, W)
        gamma: 양자화 파라미터 (B, 2 * C) - 채널별 (alpha, beta)
        """
        # gamma에서 alpha와 beta 추출
        channels = 4
        alpha = gamma[:, :channels].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = gamma[:, channels:].unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
    
        # Inverse Affine Transformation
        reconstructed = (z_hat - beta) / alpha
        return reconstructed.half()

    def decode_latent(self, latent):
        with torch.no_grad():
            recon = self.vae.decode(latent / 0.18215).sample  # shape: (B, 3, H, W), range approx [-1,1]
        recon = (recon.clamp(-1, 1) + 1) / 2  # now range [0,1]
        return recon  # torch tensor on GPU

    def calculate_xt(self, latent, timestep):

        alpha_t = self.scheduler.alphas_cumprod[timestep]
        noise = torch.randn_like(latent, dtype=torch.float16).to(latent.device)
        # alpha_t = 1 -> original(x_0) / alpha_t = 0 -> Pure noise(x_T)
        xt = torch.sqrt(alpha_t) * latent + torch.sqrt(1 - alpha_t) * noise
        return xt
        import torch

    def alpha_func(self, t_float, alpha0=1.0, alphaT=1e-5):
        """
        t_float: tensor in [0, 1]
        alpha0 : alpha at t=0 (usually 1.0)
        alphaT : alpha at t=1 (usually a small value, e.g. 1e-5)

        return: alpha(t_float), shape same as t_float
        """
        # 예: alpha(t) = alpha0^(1-t) * alphaT^t
        # => t=0일 때 alpha(0)=alpha0, t=1일 때 alpha(1)=alphaT
        return alpha0 ** (1 - t_float) * alphaT ** (t_float)
    
    import torch.nn.functional as F

    def continuous_t_embedding(self, t_float, embed_dim=320):
        """
        t_float: shape [B], each in [0,1]
        return:  shape [B, embed_dim] (positional/temporal embedding)
        """
        # 예: sinusoidal embedding (단순 버전)
        # pos = t_float.unsqueeze(1)  # shape (B,1)
        # i = torch.arange(embed_dim, device=pos.device).float()  # (embed_dim,)
        # freqs = 10000 ** (2*(i//2) / embed_dim)
        # embedding_sin = torch.sin(pos * freqs)
        # embedding_cos = torch.cos(pos * freqs)
        # combined = torch.cat([embedding_sin, embedding_cos], dim=1)[:,:embed_dim]
        # return combined

        # 여기서는 매우 단순화. t_float -> (B, embed_dim)로 확장
        return t_float.unsqueeze(1).expand(-1, embed_dim)
    
    def denoise(self, latent, start_timestep):

        # Text conditioning: Use the default placeholder prompt
        prompt = "Placeholder prompt for guidance"
        text_embeddings = self.diffusion_pipe.text_encoder(
            self.diffusion_pipe.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        )[0]

        ## Modified!!
        timesteps = self.scheduler.timesteps[1000-torch.round(1000*start_timestep):] # if start_timestep = 5 / [5,4,3,2,1]
        
        for t in timesteps:
            with torch.no_grad():
                noise_pred = self.diffusion_pipe.unet(
                    latent,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample ## Modified
        return latent
    
    def alpha_func_linearinterp(self, t_float):
      """
      t_float: shape (B,) in [0,1]
      alpha_array: shape (N,) - e.g. N=1000

      반환: alpha(t_float), shape (B,)
          piecewise linear로 alpha_array를 보간
      """
      device = t_float.device
      alpha_array = self.scheduler.alphas_cumprod.to(device)  # device = t_float.device
      N = alpha_array.shape[0]  # e.g. 1000

      # 1) 실수 인덱스
      idx = t_float * (N-1)     # shape (B,)

      # 2) 아래서 floor, ceil 구해서 보간
      idx_floor = torch.floor(idx).long()        # (B,)
      idx_ceil  = torch.clamp(idx_floor+1, max=N-1)  # (B,)

      # 3) 보간 비율
      r = idx - idx_floor.float()  # (B,)

      # 4) alpha_array는 (N,)이므로 gather로 뽑아쓰거나 fancy-index
      #   alpha_floor, alpha_ceil : (B,)
      alpha_floor = alpha_array[idx_floor]
      alpha_ceil  = alpha_array[idx_ceil]

      # 5) 선형 보간
      alpha_val = alpha_floor*(1-r) + alpha_ceil*r
      return alpha_val

    def onestep_ddim(self, x_t, t_float):
        """
        x_t: (B, C, H, W)  (latent)
        t_float: (B,) in [0,1]
        unet:  callable, e.g. eps_theta = unet(x_t, t_embedding(...))
        alpha_func: function to get alpha(t)

        return: x0_approx = shape (B, C, H, W)
        """
        # 1) compute alpha(t), shape (B,)
        alpha_t = self.alpha_func_linearinterp(t_float)  # continuous schedule
        # expand to (B,1,1,1) for broadcast
        alpha_t_4d = alpha_t.view(-1,1,1,1)
        sqrt_alpha_t = alpha_t_4d.sqrt()
        sqrt_1_minus_alpha_t = (1. - alpha_t_4d).sqrt()

        # Text conditioning: Use the default placeholder prompt
        prompt = "Placeholder prompt for guidance"
        text_embeddings = self.diffusion_pipe.text_encoder(
            self.diffusion_pipe.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        )[0]
        
        with torch.no_grad():
                eps_theta = self.diffusion_pipe.unet(
                    x_t,
                    t_float,
                    encoder_hidden_states=text_embeddings
                ).sample

        # 3) x0_approx = (x_t - sqrt(1 - alpha_t)* eps) / sqrt(alpha_t)
        x0_approx = (x_t - sqrt_1_minus_alpha_t * eps_theta) / sqrt_alpha_t
        return x0_approx