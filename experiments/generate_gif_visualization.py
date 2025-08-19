#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
獨立的GIF可視化生成腳本
用於為訓練好的模型生成動態可視化
"""

import os
import sys
import argparse

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.dynamic_visualization import main as generate_visualizations

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description='Generate dynamic GIF visualizations for trained RL models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 模型和數據配置
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--model-path', type=str, default='./models',
                           help='訓練模型參數路徑')
    model_group.add_argument('--episode', type=int, default=100,
                           help='要載入的模型episode編號')
    
    # 環境配置
    env_group = parser.add_argument_group('環境配置')
    env_group.add_argument('--ac-config', type=int, default=2, choices=[2, 6, 10],
                          help='每個服務器的AC數量配置')
    
    # 數據配置
    data_group = parser.add_argument_group('數據配置')
    data_group.add_argument('--csv', type=str, default=None,
                          help='包含任務數據的CSV文件路徑')
    
    # 輸出配置
    output_group = parser.add_argument_group('輸出配置')
    output_group.add_argument('--output-path', type=str, default='./visualization_results',
                            help='可視化結果保存路徑')
    
    # 方法選擇
    method_group = parser.add_argument_group('方法選擇')
    method_group.add_argument('--methods', nargs='+',
                            default=['MDPT4JS-HA', 'MDPT4JS', 'ACT4JS', 'AC', 'DQN', 'First Fit', 'Random Fit'],
                            choices=['MDPT4JS-HA', 'MDPT4JS', 'ACT4JS', 'AC', 'DQN', 'First Fit', 'Random Fit'],
                            help='要生成可視化的方法列表')
    method_group.add_argument('--single-method', type=str, default=None,
                            choices=['MDPT4JS-HA', 'MDPT4JS', 'ACT4JS', 'AC', 'DQN', 'First Fit', 'Random Fit'],
                            help='只生成單一方法的可視化')
    
    return parser.parse_args()

def validate_arguments(args):
    """驗證命令行參數"""
    errors = []
    
    # 檢查模型路徑
    if not os.path.exists(args.model_path):
        errors.append(f"模型路徑不存在: {args.model_path}")
    
    # 檢查CSV文件
    if args.csv and not os.path.exists(args.csv):
        errors.append(f"CSV文件不存在: {args.csv}")
    
    # 檢查AC配置對應的模型目錄
    model_config_path = os.path.join(args.model_path, f"ac{args.ac_config}")
    if not os.path.exists(model_config_path):
        errors.append(f"找不到AC配置{args.ac_config}的模型目錄: {model_config_path}")
    
    if errors:
        print(" 參數驗證失敗:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    return True

def main():
    """主函數"""
    # 解析參數
    args = parse_arguments()
    
    # 驗證參數
    if not validate_arguments(args):
        return 1
    
    # 如果指定了單一方法，覆蓋方法列表
    if args.single_method:
        args.methods = [args.single_method]
    
    print(" 動態可視化生成器")
    print("=" * 60)
    print(f" 模型路徑: {args.model_path}")
    print(f" AC配置: {args.ac_config} (總容器數: {500 * args.ac_config})")
    print(f" 數據來源: {args.csv or '隨機生成數據'}")
    print(f" 輸出路徑: {args.output_path}")
    print(f" 模型Episode: {args.episode}")
    print(f" 目標方法: {', '.join(args.methods)}")
    print("=" * 60)
    
    # 臨時修改sys.argv以傳遞參數給dynamic_visualization模組
    original_argv = sys.argv
    sys.argv = [
        'dynamic_visualization.py',
        '--model-path', args.model_path,
        '--output-path', args.output_path,
        '--ac-config', str(args.ac_config),
        '--episode', str(args.episode),
        '--methods'] + args.methods
    
    if args.csv:
        sys.argv.extend(['--csv', args.csv])
    
    try:
        # 調用可視化生成函數
        results = generate_visualizations()
        
        if results:
            print("\n 可視化生成完成！")
            print(f" 結果保存在: {args.output_path}")
            print(f" GIF文件保存在: {args.output_path}/gifs/")
            
            # 顯示生成的文件
            gif_dir = os.path.join(args.output_path, "gifs")
            if os.path.exists(gif_dir):
                gif_files = [f for f in os.listdir(gif_dir) if f.endswith('.gif')]
                if gif_files:
                    print("\n 生成的GIF文件:")
                    for gif_file in sorted(gif_files):
                        file_path = os.path.join(gif_dir, gif_file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                        print(f"    {gif_file} ({file_size:.1f} MB)")
            
            return 0
        else:
            print("\n 沒有生成任何可視化結果")
            return 1
            
    except Exception as e:
        print(f"\n 生成可視化時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 恢復原始sys.argv
        sys.argv = original_argv

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)