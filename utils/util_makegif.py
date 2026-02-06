import torch
import os
import numpy as np
from PIL import Image, ImageDraw
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

    def record_frame(self, model, epoch):
        """이미지 상단에 여백을 만들어 Epoch 정보를 표시"""
        with torch.no_grad():
            # 1. 이미지 생성 (기존 동일)
            generated = model(self.fixed_z)
            if self.configs['model']['activation'] == 'tanh':
                generated = (generated + 1) / 2
            generated = generated.clamp(0, 1)
            
            # 2. 그리드 만들기 (기존 동일)
            grid_img = make_grid(generated, nrow=int(np.sqrt(self.num_samples)), padding=2)
            
            # 3. Tensor -> PIL Image 변환 (기존 동일)
            ndarr = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            

            header_height = 25  # 텍스트가 들어갈 높이 (픽셀 단위)
            width, height = im.size
            
            # 새로운 빈 캔버스 생성 (너비는 그대로, 높이는 헤더만큼 추가)
            # 배경색은 검정(0, 0, 0)으로 설정 (흰색 원하면 (255, 255, 255))
            new_im = Image.new('RGB', (width, height + header_height), (0, 0, 0))
            
            # 기존 그리드 이미지를 헤더 아래쪽(0, header_height) 좌표에 붙여넣기
            new_im.paste(im, (0, header_height))
            
            # ---------------------------------------------------------
            # [텍스트 그리기] 이제 이미지가 아닌 상단 여백에 그림
            # ---------------------------------------------------------
            draw = ImageDraw.Draw(new_im)
            text = f"Epoch: {epoch}"
            
            # (5, 5) 위치는 이제 검은색 여백 위이므로 사진을 가리지 않음
            draw.text((5, 5), text, fill=(255, 255, 255)) # 흰색 텍스트

            self.frames.append(new_im)

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