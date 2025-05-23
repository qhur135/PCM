import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import os
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import numpy as np

import logging


from audio_k_dataset import data_loader
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd


# 데이터 로딩
text_path = '/home/ryumj/cupcake_backup/multimodal_text.txt'
txt_list = pd.read_csv(text_path, sep='\t')

# 열 이름
score_columns = ['V', 'A']

# 샘플링할 데이터 수
num_samples = 10000

# 스코어 열에서 범위 자동으로 찾기
bins = {}
for col in score_columns:
    min_val = txt_list[col].min()
    max_val = txt_list[col].max()
    # 4개의 구간으로 나누기
    bins[col] = np.linspace(min_val, max_val, 5)

# 스코어를 기준으로 데이터를 고르게 샘플링하는 함수
def sample_within_bins(df, bins, num_samples):
    sampled_df = pd.DataFrame()
    for col, bin_edges in bins.items():
        # 각 bin의 경계값에 따라 그룹화
        df[col + '_bin'] = pd.cut(df[col], bins=bin_edges, include_lowest=True)
        
        # 각 bin에서 샘플 수를 균등하게 뽑기
        bin_sampled_dfs = []
        num_bins = len(bin_edges) - 1  # 구간의 수 (4개 구간)
        num_samples_per_bin = num_samples // num_bins
        
        for bin_name, group in df.groupby(col + '_bin'):
            # 각 bin에서 샘플을 추출 (부족한 경우를 대비하여 replace=True 사용)
            if len(group) > 0:  # 그룹이 비어 있지 않은 경우에만 샘플링
                sampled_bin = group.sample(n=min(num_samples_per_bin, len(group)), replace=False)
                bin_sampled_dfs.append(sampled_bin)
        
        # 최종적으로 합치기
        bin_sampled_df = pd.concat(bin_sampled_dfs)
        sampled_df = pd.concat([sampled_df, bin_sampled_df]).drop_duplicates().reset_index(drop=True)

    # 최종 샘플 수를 맞추기
    if len(sampled_df) > num_samples:
        sampled_df = sampled_df.sample(n=num_samples)
    
    return sampled_df

# 데이터에서 샘플링
sampled_txt_list = sample_within_bins(txt_list, bins, num_samples)

# 결과 확인
print(f"Sampled data shape: {sampled_txt_list.shape}")

# 샘플링된 데이터 저장
sampled_txt_list.to_csv('/home/ryumj/cupcake_backup/multimodal_text_sampled.txt', sep='\t', index=False)