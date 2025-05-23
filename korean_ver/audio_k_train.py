import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import os
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import numpy as np

import logging
import gc

# 로그 생성
logger = logging.getLogger()

# 로그의 출력 기준 설정
logger.setLevel(logging.INFO)

# log 출력
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.FileHandler('vad_audio_k_w2v2k.log')
logger.addHandler(file_handler)

## Loss function
def MSELoss(pred_VADs, VADs):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.MSELoss()
    loss_val = loss(pred_VADs, VADs.float())
    return loss_val

## 모델 저장하기
def SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model_w2v2k.bin'))

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

def concordance_correlation_coefficient(observed, predicted):
    mean_observed = np.mean(observed)
    mean_predicted = np.mean(predicted)
    #print("ob :", observed)
    #print("pr : ", predicted)
    covariance = np.cov(observed, predicted, bias=True)[0, 1]

    obs_variance = np.var(observed, ddof=1)
    pred_variance = np.var(predicted, ddof=1)

    CCC = 2 * covariance / (obs_variance + pred_variance + (mean_observed - mean_predicted)**2)

    return CCC

def CalConcordanceCorrelation(model, dataloader):
    model.eval()

    logits_v, logits_a, logits_d = [], [], []
    v, a, d = [], [], []

    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader,mininterval=10)):

            """Prediction"""
            batch_audio,batch_sr, batch_vad = data
            
            if isinstance(batch_audio, tuple):
                pitch_audio = batch_audio[0].cuda()
                batch_audio = batch_audio[1]
            else:
                pitch_audio = torch.tensor([0]).cuda()
                
            """cuda 할당"""
            batch_audio = batch_audio.cuda()
            batch_sr = batch_sr.cuda()
            batch_vad = batch_vad.cuda()

            """Prediction"""
            pred_logits = model(pitch_audio, batch_audio)

            if torch.isnan(pred_logits).any():
                print("NaN detected in predictions. Skipping this batch ccc")
                continue
            if pred_logits is None:
                print("Model output is None. Skipping this batch ccc")
                continue  # Skip this batch if model output is None
            
            #print("pred_logits")
            #print(pred_logits)
            #pred_v = pred_logits[:, 0].cpu().numpy()
            #pred_a = pred_logits[:, 1].cpu().numpy()
            #pred_d = pred_logits[:, 2].cpu().numpy()

            pred_v = pred_logits[:, 0].cpu().numpy()
            pred_a = pred_logits[:, 1].cpu().numpy()


            #batch_v = batch_vad[:, 0].cpu().numpy()
            #batch_a = batch_vad[:, 1].cpu().numpy()
            #batch_d = batch_vad[:, 2].cpu().numpy()

            batch_v = batch_vad[:, 0].cpu().numpy()
            batch_a = batch_vad[:, 1].cpu().numpy()

            #logits_v.append(pred_v)
            logits_v.append(pred_v)
            logits_a.append(pred_a)

            #v.append(batch_v)
            v.append(batch_v)
            a.append(batch_a)

    # Convert to NumPy arrays

    logits_v = np.concatenate(logits_v)
    logits_a = np.concatenate(logits_a)
    #logits_d = np.concatenate(logits_d)
    v = np.concatenate(v)
    a = np.concatenate(a)
    #d = np.concatenate(d)

    # Calculate Concordance correlation coefficient
    ccc_V = concordance_correlation_coefficient(logits_v, v)
    ccc_A = concordance_correlation_coefficient(logits_a, a)
    #ccc_d = concordance_correlation_coefficient(logits_d, d)

    return  ccc_V, ccc_A #ccc_V, ccc_A, ccc_d

## 데이터 불러오기
from audio_k_dataset import data_loader
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

def _split_data(data):
    train, _ = train_test_split(data, shuffle=False, test_size=0.2) # random_state=42,,
    valid, test = train_test_split(_, shuffle=False, test_size=0.5) # random_state=42,,
    return train, valid, test
    
text_path = '/home/ryumj/cupcake_backup/multimodal_text_sampled.txt'#'_sampled/home/ryumj/cupcake_backup/multimodal_text.txt'
txt_list = pd.read_csv(text_path, sep='\t')[:100]
train, val, test = _split_data(txt_list)

train_dataset = data_loader(train)
dev_dataset = data_loader(val)
test_dataset = data_loader(test)

#train_dataset, dev_dataset, test_dataset = data_loader('train').total

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=train_dataset.collate_fn) # num_workers=4,
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True,  collate_fn=dev_dataset.collate_fn) #num_workers=4,
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=test_dataset.collate_fn) #num_workers=4,

## 모델 불러오기
from audio_k_model import AudioClassifier
vad_model = AudioClassifier().cuda()

## 학습 코드

# 학습 코드의 흐름
# logit과 label을 통해 loss값을 계산하고
# 계산된 loss값을 optimizer로 학습시키면 됨.
"""하이퍼 파라미터들"""
training_epochs = 100
max_grad_norm = 10
lr = 1e-05
num_training_steps = len(train_dataset)*training_epochs
num_warmup_steps = len(train_dataset)
optimizer = torch.optim.AdamW(vad_model.parameters(), lr=lr) # eps = 1e-06, weight_decay=0.01
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

best_score= 0
save_path = './vad_k_model'

# AMP 설정 추가
scaler = torch.cuda.amp.GradScaler()

logger.info("#############학습 시작#############")

for epoch in range(training_epochs):
    vad_model.train()
    loss = 0

    for i_batch, data in enumerate(tqdm(train_dataloader,mininterval=10)):
        try:
            batch_audio,batch_sr, batch_vad = data
            
            if isinstance(batch_audio, tuple):
                pitch_audio = batch_audio[0].cuda()
                batch_audio = batch_audio[1]
                if batch_audio is None or pitch_audio is None:
                    print("pitch값없음. 이 배치 건너뜀.")
                    continue  # 현재 배치를 건너뜁니다.
                    
            else:
                pitch_audio = torch.tensor([0]).cuda()
                
            """cuda 할당"""
            batch_audio = batch_audio.cuda()
            batch_sr = batch_sr.cuda()
            batch_vad = batch_vad.cuda()
            
            
            """Prediction"""
            #with torch.cuda.amp.autocast():  # Mixed precision을 적용하여 메모리 사용량 줄이기
            pred_logits = vad_model(pitch_audio, batch_audio)
            if torch.isnan(pred_logits).any():
                print("NaN detected in predictions. Skipping this batch")
                continue
            if pred_logits is None:
                print("Model output is None. Skipping this batch")
                continue  # Skip this batch if model output is None
            
            """Loss claculation & training"""
            loss_val = MSELoss(pred_logits, batch_vad)
            
            
            loss += loss_val.item()

            loss_val.backward(retain_graph=True)#
            torch.nn.utils.clip_grad_norm_(vad_model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            #torch.cuda.empty_cache()
            gc.collect()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error(f"Out of memory at batch {i_batch}: {e}")
                torch.cuda.empty_cache()  # 캐시 비우기
                print(f"Memory cleared at batch {i_batch}.")
            else:
                raise e  # 다른 오류는 다시 발생시킵니다
        except Exception as e:
            print(f"An error occurred: {e}. Skipping this batch.")
            continue  # Skip this batch if an exception occurs

    """Dev & Test evaluation"""
    loss = loss/len(train_dataloader)

    vad_model.eval()
    logger.info("Train MSE Loss : {}".format(loss))
    #dev_cccV, dev_cccA, dev_cccD = CalConcordanceCorrelation(vad_model, dev_dataloader)
    dev_cccV, dev_cccA = CalConcordanceCorrelation(vad_model, dev_dataloader)

    logger.info("\n# -- Training epoch : {:.0f} -- #".format(epoch))
    #logger.info("\nDev avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f} ".format(dev_cccV, dev_cccA, dev_cccD))
    logger.info("\nDev avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f} ".format(dev_cccV, dev_cccA))

    #dev_avg = (dev_cccV+dev_cccA+dev_cccD)/3
    dev_avg = (dev_cccV+dev_cccA)/2

    """Best Score & Model Save"""
    curr_score = dev_avg
    if curr_score > best_score:
        best_score = curr_score

        #test_cccV, test_cccA, dev_cccD = CalConcordanceCorrelation(vad_model, test_dataloader)
        test_cccV, test_cccA = CalConcordanceCorrelation(vad_model, test_dataloader)

        SaveModel(vad_model, save_path)
        #logger.info("\nTest avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f} ".format(test_cccV, test_cccA, dev_cccD))
        logger.info("\nTest avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f} ".format( test_cccV, test_cccA))