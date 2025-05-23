import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.nn import CTCLoss
import torch.nn.functional as F

from iemocap_audio_model import AudioClassifier
#from iemocap_audio_dataset import data_loader
from audio_parsing_dataset import data_loader
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC

import os 
import torch.nn as nn
import numpy as np

import logging

import argparse
from argparse import ArgumentParser
import json
import gc

from scipy.signal import resample

class Trainer:
    def __init__(self,args):
        
        self._check_args(args)
        self.args = args
        
        self.logger = None
        self.vad_model = AudioClassifier(self.args).cuda()
        #self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large',padding_side='left')
        if args.task in ['ctc_normal']:
            self.audio_model = Wav2Vec2ForCTC.from_pretrained(self.args.audio_model_name).cuda()
        else:
            self.audio_model = Wav2Vec2Model.from_pretrained(self.args.audio_model_name, mask_feature_length=10).cuda() #Wav2Vec2Model.
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        self.ctc_loss = CTCLoss(zero_infinity=True)        

        self.inferenced_value = []
        
        self.best_score= 0
        self.save_path = self.args.load_model_path
        self.logger_path = self.args.logger_path
    
    def _check_args(self, args):
        assert args.task in ['normal','temporal_normal','temporal_intonation_and_wav2vec2','interpolated_intonation_and_wav2vec2','msp_normal','msp_intonation_and_wav2vec2','contour_image_vad','audio_parsing','audio_parsing_probing','ctc_normal','contour_image_and_ctc','intonation_contour','normal_with_text','audio_parsing_with_text','audio_parsing_with_text_concat','intonation_contour_with_text','intonation_and_wav2vec2','mfcc_wav2vec2','double_wav2vec2']
        assert args.dataset in ['iemocap', '멀티모달 영상','msp_podcast']
                
    def set_logger(self,):
        # 로그 생성
        self.logger = logging.getLogger()

        # 로그의 출력 기준 설정
        self.logger.setLevel(logging.INFO)

        # log 출력
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)

        # log를 파일에 출력
        file_handler = logging.FileHandler(self.logger_path)
        self.logger.addHandler(file_handler)

    ## Loss function
    def MSELoss(self, pred_VADs, VADs):
        """
            pred_outs: [batch, clsNum]
            labels: [batch]
        """
        loss = nn.MSELoss()
        #print("여기까지는 들어오냐능")
        loss_val = loss(pred_VADs, VADs.float())
        #print("MSE vad : ", VADs.float())
        #print("MSE vad : ", VADs.shape)
        #print("pred ddd: ", pred_VADs)
        #print("pred : ", pred_VADs.shape)
        #print(loss_val.shape)
        #print(loss_val)
        return loss_val       
        ## 모델 저장하기
    
    #모델 저장하기
    def SaveModel(self, model, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, 'model.bin'))

    #모델 불러오기
    def get_model(self,):
        self.vad_model.load_state_dict(torch.load(self.save_path))
        self.vad_model.eval()
    
    # CCC score 계산            
    def concordance_correlation_coefficient(self, observed, predicted):      
        # CCC 계산
        mean_observed = np.mean(observed)
        mean_predicted = np.mean(predicted)
        #print("ob :", observed.shape)
        #print("pr : ", predicted.shape)
        covariance = np.cov(observed, predicted, bias=True)[0, 1]
        obs_variance = np.var(observed, ddof=1)
        pred_variance = np.var(predicted, ddof=1)
        CCC = 2 * covariance / (obs_variance + pred_variance + (mean_observed - mean_predicted)**2)
        return CCC
    # inference
    def CalConcordanceCorrelation(self, model, dataloader):
        model.eval()
        
        logits_v, logits_a, logits_d = [], [], []
        v, a, d = [], [], []
        
        with torch.no_grad():
            for i_batch, data in enumerate(tqdm(dataloader,mininterval=10)):

                """Prediction"""
                dialog_id, speaker, batch_padding_tokens, batch_attention_mask, batch_audio,batch_sr, batch_vad = data

                """cuda 할당"""
                if batch_padding_tokens is not None:
                #print("여기 안 들어옴?")
                    batch_padding_tokens = batch_padding_tokens.cuda()
                    batch_attention_mask = batch_attention_mask.cuda()    
                if  batch_attention_mask is not None:
                    batch_attention_mask = batch_attention_mask.cuda()
                
                pitch_audio = None
                if isinstance(batch_audio, tuple):
                    pitch_audio = batch_audio[0].cuda()
                    batch_audio = batch_audio[1]
                if self.args.task == "mfcc_wav2vec2":
                    #print("pitch_audio : ", pitch_audio.shape)
                    pitch_audio = pitch_audio.squeeze(0).squeeze(0)
                batch_audio = batch_audio.cuda()
                batch_sr = batch_sr.cuda()
                batch_vad = batch_vad.cuda()

                #print("어텐션 마스크 : ", batch_attention_mask.is_cuda)
                #print("패딩된 토큰 : ", batch_padding_tokens.is_cuda)
                """Prediction"""  
                pred_logits = model(pitch_audio, batch_audio,batch_padding_tokens,batch_attention_mask)
                
                #print("pred_logits shape : ", pred_logits.shape)
                # 모델이 None을 반환하는 경우를 처리합니다.
                if pred_logits is None:
                    continue
                if self.args.task in ["audio_parsing_with_text","contour_image_vad"]:
                    pred_logits = pred_logits.unsqueeze(dim=0)
                if self.args.task in ["ctc_normal"]:
                    pred_logits = pred_logits[1]
                elif self.args.task in ["contour_image_and_ctc"]:
                    pred_logits = pred_logits[1].unsqueeze(dim=0)
                
                #print("pred_logits : ", pred_logits.shape)
                #print("pred_logit : ", pred_logits.shape)
                pred_v = pred_logits[:, 0].cpu().numpy()
                pred_a = pred_logits[:, 1].cpu().numpy()
                pred_d = pred_logits[:, 2].cpu().numpy()

                #print("batch_vad : ", batch_vad.shape)
                batch_v = batch_vad[:, 0].cpu().numpy()
                batch_a = batch_vad[:, 1].cpu().numpy() 
                batch_d = batch_vad[:, 2].cpu().numpy()
                
                logits_v.append(pred_v)
                logits_a.append(pred_a)
                logits_d.append(pred_d)

                v.append(batch_v) 
                a.append(batch_a)
                d.append(batch_d)  

        # Convert to NumPy arrays
        logits_v = np.concatenate(logits_v)
        logits_a = np.concatenate(logits_a)
        logits_d = np.concatenate(logits_d)
        v = np.concatenate(v)
        a = np.concatenate(a)
        d = np.concatenate(d)    

        # Calculate Concordance correlation coefficient
        ccc_V = self.concordance_correlation_coefficient(v, logits_v)
        ccc_A = self.concordance_correlation_coefficient(a, logits_a)
        ccc_d = self.concordance_correlation_coefficient(d, logits_d)

        return ccc_V, ccc_A, ccc_d
    
    def set_parameters(self,):
        """사전학습 모델 얼리기"""
        self.vad_model.freeze_prameters()
        
        """하이퍼 파라미터들"""
        self.training_epochs = self.args.epoch
        self.max_grad_norm = self.args.max_grad_norm
        self.lr = self.args.learning_rate
        self.num_training_steps = len(self.train_dataset)*self.training_epochs
        self.num_warmup_steps = len(self.train_dataset)
        self.optimizer = torch.optim.AdamW(self.vad_model.parameters(), lr=self.lr) # eps = 1e-05, weight_decay=0.01
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
    
    def _split_data(self,data): 
        train, _ = train_test_split(data, shuffle=self.args.train_shuffle,test_size=self.args.val_ratio) # ,random_state=42, 
        valid, test = train_test_split(_, shuffle=self.args.train_shuffle,test_size=self.args.test_ratio) # ,random_state=42,          
        return train, valid, test #0.8:0.1:0.1    
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    # 학습 시키기
    def train(self,):

        #--- 데이터셋 설정 ---#
        
        if self.args.dataset == "iemocap":
            df = pd.read_csv(self.args.text_cap_path, sep='\t', encoding='utf-8')
          
            train = df[df['name'].str.contains('Ses01|Ses02|Ses03')]#[:500]
            val = df[df['name'].str.contains('Ses04')]#[:100]
            test = df[df['name'].str.contains('Ses05')]#[:100]
            
            #train, val, test = self._split_data(df)
        elif self.args.dataset == "msp_podcast":
            df = pd.read_csv(self.args.text_msp_path, sep=',')
            train = df[df['Split_Set'] == 'Train']#[:500]
            test = df[df['Split_Set'] == 'Test1']#[:100]
            val = df[df['Split_Set'] == 'Development']#[:100]
                            
            print(train[:5])
        
        self.train_dataset = data_loader(train,self.args)
        self.dev_dataset = data_loader(val,self.args)
        self.test_dataset = data_loader(test,self.args)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=self.args.train_shuffle,  collate_fn=self.train_dataset.collate_fn) #num_workers=4,
        self.dev_dataloader = DataLoader(self.dev_dataset, batch_size=self.args.eval_batch_size, shuffle=self.args.train_shuffle,  collate_fn=self.dev_dataset.collate_fn) #num_workers=4,
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.eval_batch_size, shuffle=self.args.train_shuffle, collate_fn=self.test_dataset.collate_fn) #num_workers=4,
        
        self.set_logger()
        self.set_parameters()
        self.set_seed(self.args.seed)
        
        #model_path = './vad_iemocap_audio_new_model/model.bin'
        #self.vad_model.load_state_dict(torch.load(model_path))

        self.logger.info("#############학습 시작#############")

        for epoch in range(self.training_epochs):
            self.vad_model.train()
            loss = 0
            
            for i_batch, data in enumerate(tqdm(self.train_dataloader,mininterval=10)):
                try:
                    dialog_id, speaker, batch_padding_tokens, batch_attention_mask, batch_audio,batch_sr, batch_vad = data

                    """cuda 할당"""
                    if batch_padding_tokens is not None:
                        #print("들어오지?")
                        #print("여기 안 들어와?")
                        batch_padding_tokens = batch_padding_tokens.cuda()
                        batch_attention_mask = batch_attention_mask.cuda()
                    if  batch_attention_mask is not None:
                        batch_attention_mask = batch_attention_mask.cuda()
                    pitch_audio = None
                    if isinstance(batch_audio, tuple):
                        pitch_audio = batch_audio[0].cuda()
                        batch_audio = batch_audio[1]
                    if self.args.task == "mfcc_wav2vec2":
                        #print("pitch_audio : ", pitch_audio.shape)
                        pitch_audio = pitch_audio.squeeze(0).squeeze(0)
                    if self.args.task == "ctc_normal":
                        batch_padding_tokens = batch_padding_tokens.cuda()
                                         
                    batch_audio = batch_audio.cuda()
                    #batch_audio = whole_audio.cuda()   
                    batch_sr = batch_sr.cuda()
                    batch_vad = batch_vad.cuda()

                    """Prediction"""
                    pred_logits = self.vad_model(pitch_audio, batch_audio,batch_padding_tokens,batch_attention_mask)
                    
                    if pred_logits is None:
                        print("Model output is None. Skipping this batch")
                        continue  # Skip this batch if model output is None
                    
                    """Loss claculation & training"""
                    if self.args.task in ['ctc_normal','contour_image_and_ctc']:
                        ctc_logits, vad_logits = pred_logits
                        # batch_size와 seq_length 가져오기
                        batch_size = ctc_logits.size(0)
                        seq_length = ctc_logits.size(1)

                        # input_lengths와 target_lengths를 시퀀스 길이와 마스크를 기반으로 계산
                        input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long).cuda()

                        labels_mask = batch_padding_tokens >= 0
                        target_lengths = labels_mask.sum(-1)

                        # CTC Loss 계산
                        ctc_logits = torch.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)  # CTC Loss는 log probabilities를 기대하며, 차원 변환이 필요합니다

                        # 필요한 설정을 직접 정의
                        blank_token_id = 0  # 예시로, pad_token_id가 0이라고 가정
                        ctc_loss_reduction = 'mean'  # 또는 'sum'
                        ctc_zero_infinity = True

                        with torch.backends.cudnn.flags(enabled=False):  # CUDNN 비결정성 제어
                            ctcloss = F.ctc_loss(
                                ctc_logits,
                                batch_padding_tokens.masked_select(labels_mask),
                                input_lengths,
                                target_lengths,
                                blank=blank_token_id,
                                reduction=ctc_loss_reduction,
                                zero_infinity=ctc_zero_infinity
                            )
                        
                        vadloss = self.MSELoss(vad_logits, batch_vad)
                        
                        
                        #print("ctc_loss : ", ctcloss)
                        #print("vad_loss : ", vadloss)
                        
                        loss_val = self.args.ctc_a * ctcloss + vadloss
                    else:
                        loss_val = self.MSELoss(pred_logits, batch_vad)
                    
                    #if loss_val is None:
                    #    continue  # Skip loss calculation and training if loss is None
                    
                    # NaN 값 검사
                    if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
                        print("NaN or Inf detected in loss_val")
                        
                    loss += loss_val.item()
                    
                    loss_val.backward(retain_graph=True)#
                    torch.nn.utils.clip_grad_norm_(self.vad_model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    #torch.cuda.empty_cache()
                    gc.collect()
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        self.logger.error(f"Out of memory at batch {i_batch}: {e}")
                        torch.cuda.empty_cache()  # 캐시 비우기
                        print(f"Memory cleared at batch {i_batch}.")
                    else:
                        raise e  # 다른 오류는 다시 발생시킵니다
                except Exception as e:
                    print(f"An error occurred: {e}. Skipping this batch.")
                    continue  # Skip this batch if an exception occurs            
            #torch.cuda.empty_cache()
            loss = loss/len(self.train_dataloader)    
            """Dev & Test evaluation"""
            
            self.vad_model.eval()
            self.logger.info("Train MSE Loss : {}".format(loss))
            dev_cccV, dev_cccA, dev_cccD = self.CalConcordanceCorrelation(self.vad_model, self.dev_dataloader)
            
            self.logger.info("\n# -- Training epoch : {:.0f} -- #".format(epoch))
            self.logger.info("\nDev avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f} ".format(dev_cccV, dev_cccA, dev_cccD))
            
            dev_avg = (dev_cccV+dev_cccA+dev_cccD)/3
            
            """Best Score & Model Save"""
            curr_score = dev_avg
            if curr_score > self.best_score: 
                self.best_score = curr_score
                
                test_cccV, test_cccA, dev_cccD = self.CalConcordanceCorrelation(self.vad_model, self.test_dataloader)
                
                self.SaveModel(self.vad_model, self.save_path)     
                self.logger.info("\nTest avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f} ".format(test_cccV, test_cccA, dev_cccD))
    # python cupcake_backup/iemocap_code/iemocap_audio_train.py --config /home/ryumj/cupcake_backup/configs/audio.json"

    def inference(self):
        # 데이터셋 설정
        if self.args.dataset == "iemocap":
            df = pd.read_csv(self.args.text_cap_path, sep='\t', encoding='utf-8')
            _, _, test = self._split_data(df)
        elif self.args.dataset == "msp_podcast":
            df = pd.read_csv(self.args.text_msp_path, sep=',', encoding='utf-8')
            _, _, test = self._split_data(df)
        self.test_dataset = data_loader(test, self.args)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.eval_batch_size, shuffle=False, collate_fn=self.test_dataset.collate_fn)
        
        self.set_logger()
        self.set_seed(self.args.seed)
        
        self.get_model()
        
        self.logger.info("#############Inference 시작#############")
        test_cccV, test_cccA, test_cccD = self.CalConcordanceCorrelation(self.vad_model, self.test_dataloader)
        self.logger.info("Test avg concordance_correlation_coefficient for V: {:.4f}, A: {:.4f}, D: {:.4f}".format(test_cccV, test_cccA, test_cccD))
def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        args = json.load(config_file)
    args = argparse.Namespace(**args)

    emotion_trainer = Trainer(args)
    emotion_trainer.train()#.inference()#

if __name__ == "__main__":
    main()