import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# 添加 Matcha-TTS 路径
current_dir = os.getcwd()
matcha_path = os.path.join(current_dir, 'third_party/Matcha-TTS')
if matcha_path not in sys.path:
    sys.path.append(matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description='CosyVoice TTS 命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 使用默认设置(预训练音色模式)
  %(prog)s --input_dir input --output_dir output
  
  # 查看所有可用音色
  %(prog)s --list_speakers
  
  # 使用指定音色和语速
  %(prog)s --input_dir input --output_dir output --speaker "英文女" --speed 1.2
  
  # 使用3s极速复刻模式
  %(prog)s --mode "3s极速复刻" --prompt_audio voice.wav --prompt_text "示例文本" --input_dir input --output_dir output
  
  # 使用跨语种复刻模式
  %(prog)s --mode "跨语种复刻" --prompt_audio voice.wav --input_dir input --output_dir output
  
  # 使用自然语言控制模式
  %(prog)s --mode "自然语言控制" --instruct_text "控制指令文本" --input_dir input --output_dir output
        '''
    )
    
    # 创建参数组便于管理
    required = parser.add_argument_group('必需参数')
    optional = parser.add_argument_group('可选参数')
    mode_specific = parser.add_argument_group('特定模式参数')
    
    # 必需参数
    required.add_argument('--input_dir', type=str, default='input',
                       help='输入文本文件目录路径 (默认: input)')
    required.add_argument('--output_dir', type=str, default='output',
                       help='输出音频文件目录路径 (默认: output)')
    
    # 可选参数
    optional.add_argument('--mode', type=str, default='预训练音色',
                       choices=['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制'],
                       help='推理模式 (默认: 预训练音色)')
    optional.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice-300M',
                       help='预训练模型路径 (默认: pretrained_models/CosyVoice-300M)')
    optional.add_argument('--speaker', type=str, default='中文女',
                       choices=['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女'],
                       help='预训练音色名称 (默认: 中文女)')
    optional.add_argument('--stream', action='store_true',
                       help='是否使用流式推理 (默认: False)')
    optional.add_argument('--speed', type=float, default=1.0,
                       help='语速调节, 范围0.5-2.0 (默认: 1.0)')
    optional.add_argument('--seed', type=int, default=1234,
                       help='随机种子 (默认: 1234)')
    optional.add_argument('--list_speakers', action='store_true',
                       help='列出所有可用的预训练音色')
    
    # 特定模式参数
    mode_specific.add_argument('--prompt_audio', type=str,
                            help='prompt音频文件路径 (3s极速复刻/跨语种复刻模式需要)')
    mode_specific.add_argument('--prompt_text', type=str,
                            help='prompt文本 (3s极速复刻模式需要)')
    mode_specific.add_argument('--instruct_text', type=str,
                            help='instruct文本 (自然语言控制模式需要)')
    
    args = parser.parse_args()
    return args

def init_dirs(input_dir, output_dir):
    """初始化输入输出目录"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"输入目录 {input_dir} 不存在")
    
    output_path.mkdir(parents=True, exist_ok=True)
    return input_path, output_path

def load_prompt_audio(prompt_audio_path, prompt_sr=16000):
    """加载并处理prompt音频"""
    if not os.path.exists(prompt_audio_path):
        raise ValueError(f"Prompt音频文件 {prompt_audio_path} 不存在")
    
    prompt_speech_16k = load_wav(prompt_audio_path, prompt_sr)
    if prompt_speech_16k.abs().max() > 0.8:
        prompt_speech_16k = prompt_speech_16k / prompt_speech_16k.abs().max() * 0.8
    return prompt_speech_16k

def save_audio(audio_data, sample_rate, output_path):
    """保存音频文件"""
    if isinstance(audio_data, np.ndarray):
        audio_data = torch.from_numpy(audio_data)
    torchaudio.save(output_path, audio_data.unsqueeze(0), sample_rate)

def list_available_speakers():
    """列出所有可用的预训练音色"""
    speakers = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
    print("\n可用的预训练音色:")
    for speaker in speakers:
        print(f"- {speaker}")
    print()
    sys.exit(0)

def main():
    args = parse_args()
    
    # 如果指定了 --list_speakers，显示可用音色后退出
    if args.list_speakers:
        list_available_speakers()
    
    # 初始化目录
    input_path, output_path = init_dirs(args.input_dir, args.output_dir)
    
    # 初始化模型
    cosyvoice = CosyVoice(args.model_dir)
    if args.mode == '预训练音色' and not args.speaker:
        args.speaker = cosyvoice.list_avaliable_spks()[0]
    
    # 加载prompt音频(如果需要)
    prompt_speech = None
    if args.mode in ['3s极速复刻', '跨语种复刻'] and args.prompt_audio:
        prompt_speech = load_prompt_audio(args.prompt_audio)
    
    # 处理每个输入文件
    input_files = list(input_path.glob('*.txt'))
    for input_file in tqdm(input_files, desc="处理文件"):
        # 读取输入文本
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 设置随机种子
        set_all_random_seed(args.seed)
        
        # 根据不同模式生成音频
        if args.mode == '预训练音色':
            speech_generator = cosyvoice.inference_sft(
                text, args.speaker, stream=args.stream, speed=args.speed
            )
        elif args.mode == '3s极速复刻':
            if not prompt_speech or not args.prompt_text:
                raise ValueError("3s极速复刻模式需要提供prompt音频和文本")
            speech_generator = cosyvoice.inference_zero_shot(
                text, args.prompt_text, prompt_speech, stream=args.stream, speed=args.speed
            )
        elif args.mode == '跨语种复刻':
            if not prompt_speech:
                raise ValueError("跨语种复刻模式需要提供prompt音频")
            speech_generator = cosyvoice.inference_cross_lingual(
                text, prompt_speech, stream=args.stream, speed=args.speed
            )
        else:  # 自然语言控制
            if not args.instruct_text:
                raise ValueError("自然语言控制模式需要提供instruct文本")
            speech_generator = cosyvoice.inference_instruct(
                text, args.speaker, args.instruct_text, stream=args.stream, speed=args.speed
            )
        
        # 获取生成的音频
        for result in speech_generator:
            audio = result['tts_speech'].numpy().flatten()
            
            # 保存音频文件
            output_file = output_path / f"{input_file.stem}.wav"
            save_audio(audio, 22050, output_file)
            break  # 只保存第一个生成结果

if __name__ == '__main__':
    main()
