import os
import sys
import torch
import torchaudio
import librosa
import numpy as np
import argparse
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from tqdm import tqdm
import gc
import json
from datetime import datetime

# 添加 Matcha-TTS 路径
current_dir = os.getcwd()
matcha_path = os.path.join(current_dir, 'third_party/Matcha-TTS')
if matcha_path not in sys.path:
    sys.path.append(matcha_path)
    print(f"Added Matcha-TTS path: {matcha_path}")

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

class BatchTTSGenerator:
    def __init__(self, 
                 model_dir: str = 'pretrained_models/CosyVoice-300M',
                 device: str = 'cuda'):
        """
        初始化批量语音合成器
        Args:
            model_dir: 模型目录路径
            device: 使用的设备 ('cuda' 或 'cpu')
        """
        try:
            print(f"正在加载模型，模型路径: {model_dir}")
            self.cosyvoice = CosyVoice(model_dir)
            
            # 检查模型是否成功加载
            if not hasattr(self.cosyvoice, 'model'):
                raise RuntimeError("模型加载失败")
                
            if device == 'cpu':
                print("将模型移至CPU...")
                self.cosyvoice.model = self.cosyvoice.model.cpu()
            
            # 初始化其他参数
            self.device = device
            self.prompt_sr = 16000
            self.target_sr = 22050
            self.max_val = 0.8
            
            # 验证可用的说话人
            self.available_speakers = self.cosyvoice.list_avaliable_spks()
            print(f"可用说话人: {self.available_speakers}")
            
        except Exception as e:
            print(f"初始化模型时出错: {str(e)}")
            raise

    def process_folder(self,
                      input_dir: str,
                      output_dir: str,
                      mode: str = '预训练音色',
                      speaker: str = '中文女',
                      batch_size: int = 5,
                      parallel_size: int = 4,
                      stream: bool = False,
                      speed: float = 1.0,
                      seed: int = 1234) -> None:
        """处理指定目录及其子目录下的所有文本文件"""
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            
            # 验证输入目录
            if not input_path.exists():
                raise FileNotFoundError(f"输入目录不存在: {input_dir}")
            
            # 确保输出目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 获取所有txt文件（包括子目录）
            txt_files = list(input_path.rglob('*.txt'))
            if not txt_files:
                logging.warning(f"在 {input_dir} 及其子目录中未找到任何txt文件")
                return
                
            print(f"找到 {len(txt_files)} 个文本文件")
            
            # 验证说话人
            if speaker not in self.available_speakers:
                raise ValueError(f"无效的说话人 '{speaker}'。可用说话人: {self.available_speakers}")

            # 创建进度记录文件
            done_file = output_path / '.done'
            processed_files = {}
            if done_file.exists():
                with open(done_file, 'r', encoding='utf-8') as f:
                    processed_files = json.load(f)
                print(f"找到进度文件，已处理 {len(processed_files)} 个文件")

            # 处理文件
            for i, txt_file in enumerate(tqdm(txt_files, desc="处理文件")):
                # 获取相对路径，用于创建对应的输出目录结构
                rel_path = txt_file.relative_to(input_path)
                output_file = output_path / rel_path.parent / f"{txt_file.stem}.wav"
                
                # 检查是否已处理
                rel_path_str = str(rel_path)
                if rel_path_str in processed_files:
                    print(f"跳过已处理的文件: {rel_path_str}")
                    continue
                    
                try:
                    # 确保输出文件的目录存在
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 读取文本
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if not text:
                        print(f"跳过空文件: {rel_path_str}")
                        continue
                        
                    # 确保文本以标点符号结尾
                    if text[-1] not in '。！？.,!?':
                        text = text + '。'
                        
                    # 生成音频
                    print(f"正在处理文件: {rel_path_str}")
                    
                    # 使用预训练音色模式生成
                    for result in self.cosyvoice.inference_sft(text, speaker, stream=stream, speed=speed):
                        audio = result['tts_speech'].numpy().flatten()
                        self._save_audio(audio, output_file)
                        break  # 只保存第一个生成结果
                        
                    processed_files[rel_path_str] = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'success',
                        'output_file': str(output_file)
                    }
                    
                    # 定期清理显存
                    if (i + 1) % batch_size == 0 and self.device == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    print(f"处理文件 {rel_path_str} 时出错: {str(e)}")
                    processed_files[rel_path_str] = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'failed',
                        'error': str(e)
                    }
                
                # 更新进度文件
                with open(done_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_files, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            print(f"处理目录时出错: {str(e)}")
            raise

    @staticmethod
    def _save_audio(audio_data: np.ndarray, output_path: Path) -> None:
        """保存音频文件"""
        try:
            audio_tensor = torch.from_numpy(audio_data).float()
            torchaudio.save(str(output_path), audio_tensor.unsqueeze(0), 22050)
            print(f"已保存音频文件: {output_path}")
        except Exception as e:
            print(f"保存音频文件时出错: {str(e)}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description='批量文本转语音工具')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='输入文本文件目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出音频文件目录路径')
    parser.add_argument('--model_dir', type=str, 
                      default='pretrained_models/CosyVoice-300M',
                      help='模型目录路径')
    parser.add_argument('--device', type=str, 
                      default='cuda',
                      choices=['cuda', 'cpu'],
                      help='使用的设备')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== 批量语音合成开始 ===")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型目录: {args.model_dir}")
    print(f"使用设备: {args.device}")
    
    try:
        generator = BatchTTSGenerator(
            model_dir=args.model_dir,
            device=args.device
        )
        
        generator.process_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        
        print("=== 批量语音合成完成 ===")
        
    except Exception as e:
        print(f"=== 处理过程中出错 ===")
        print(f"错误信息: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
