import torch
import os
import numpy as np
import cv2
from PIL import Image
from torchvision.utils import make_grid
from utils.util_paths import get_output_dir


class SampleRecorder:
    def __init__(self, configs, device, save_filename="sampling_process.gif", save_dir=None, scale=4):
        self.configs = configs
        self.device = device
        self.save_filename = save_filename
        self.frames = []
        self.scale = scale
        
        # 저장 경로 미리 생성
        if save_dir is None:
            output_dir = get_output_dir(configs)
            self.save_dir = os.path.join(output_dir, "visualization")
        else:
            self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def record_step(self, x_t, t):
        """
        x_t: 현재 시점 t의 이미지 텐서 (Batch, C, H, W) [-1, 1] 범위
        t: 현재 타임스텝 (int)
        """
        # 1. 텐서 후처리: [-1, 1] -> [0, 1] -> [0, 255]
        # 계산 비용을 줄이기 위해 no_grad 안에서 수행
        with torch.no_grad():
            x_t = (x_t + 1) / 2
            x_t = x_t.clamp(0, 1)
            
            # 2. 그리드 만들기
            # nrow는 배치 사이즈의 제곱근으로 설정 (예: 16장 -> 4x4)
            grid = make_grid(x_t, nrow=int(np.sqrt(x_t.size(0))), padding=2)
            
            # 3. Tensor -> NumPy 변환
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

            # 4. 상단 여백 추가 및 텍스트 작성
            header_height = 30
            new_im = cv2.copyMakeBorder(
                ndarr,
                header_height,
                0,
                0,
                0,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            text = f"Step T: {t}"
            cv2.putText(
                new_im,
                text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            if self.scale != 1:
                new_width = int(new_im.shape[1] * self.scale)
                new_height = int(new_im.shape[0] * self.scale)
                new_im = cv2.resize(new_im, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

            self.frames.append(new_im)

    def save_gif(self, duration=100, loop=0):
        """
        모아둔 프레임을 GIF로 저장
        duration: 프레임 간 지연 시간 (ms) - 낮을수록 빠름
        """
        if not self.frames:
            print("저장할 프레임이 없습니다.")
            return

        save_path = os.path.join(self.save_dir, self.save_filename)
        
        # 첫 번째 이미지를 기준으로 저장
        pil_frames = [Image.fromarray(frame) for frame in self.frames]
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=False,
            duration=duration,
            loop=loop
        )
        print(f"Sampling GIF 저장 완료: {save_path}")