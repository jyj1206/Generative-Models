import torch
import os
import numpy as np
from PIL import Image
from torchvision.utils import make_grid


class TrainingProcessRecorder:
    def __init__(self, configs, device, num_samples=16):
        self.configs = configs
        self.device = device
        self.num_samples = num_samples
        
        # [핵심] 처음에 고정된 노이즈(z)를 딱 한 번만 만듭니다.
        # 이 z를 계속 재사용해야 이미지가 서서히 변하는 과정을 볼 수 있습니다.
        self.fixed_z = torch.randn(num_samples, configs['model']['latent_dim']).to(device)
        
        # 프레임(이미지)들을 저장할 리스트
        self.frames = []

    def record_frame(self, model):
        """현재 모델 상태로 이미지를 생성하여 프레임에 추가"""
        with torch.no_grad():
            # 1. 고정된 z로 이미지 생성
            # VAE와 GAN 모두 호출 가능하도록 model()을 직접 호출
            generated = model(self.fixed_z)
            
            # 2. Activation에 따른 범위 복구 (-1~1 -> 0~1)
            # configs["model"]인지 configs["train"]인지 본인 yaml에 맞게 확인 필수!
            if self.configs['model']['activation'] == 'tanh':
                generated = (generated + 1) / 2
            
            # 3. 0~1 사이로 자르기
            generated = generated.clamp(0, 1)
            
            # 4. 보기 좋게 그리드(4x4) 형태로 묶기
            # make_grid는 텐서들을 이어붙여 하나의 큰 이미지로 만들어줍니다.
            grid_img = make_grid(generated, nrow=int(np.sqrt(self.num_samples)), padding=2, normalize=False)
            
            # 5. Tensor(C, H, W) -> Numpy(H, W, C) -> uint8(0~255) 변환
            # PIL Image로 변환하기 위한 전처리입니다.
            ndarr = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            
            self.frames.append(im)

    def save_gif(self, filename="training_process.gif", duration=100):
        """모아둔 프레임을 GIF로 저장"""
        if not self.frames:
            print("저장할 프레임이 없습니다.")
            return
            
        save_path = os.path.join("output", self.configs["task_name"], "visualization", filename)
        
        # 첫 번째 이미지를 기준으로 나머지 이미지들을 붙여서 저장
        self.frames[0].save(
            save_path,
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=duration, # 프레임당 머무는 시간 (ms)
            loop=0
        )
        print(f"GIF 저장 완료: {save_path}")