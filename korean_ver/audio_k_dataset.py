import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from transformers import Wav2Vec2Model, Wav2Vec2Processor
#from kobert_tokenizer import KoBERTTokenizer
import torch
import torchaudio
import librosa
from sklearn.model_selection import train_test_split
import parselmouth
from parselmouth.praat import call
import numpy as np
import re
from tqdm import tqdm
import glob


# -- audio pitch class -- #
class Intonation_features:
    def __init__(self,sound):
        #self.args = args
        
        self.sound = sound 
        self.pitch_tiers = None
        self.total_duration = None
        self.pitch_point = []
        self.time_point = []
        
        self.pd = []
        self.pt = []
        self.ps = []
        self.pr = []
        
    def get_pitch_tiers(self,):
        manipulation = call(self.sound, "To Manipulation", 0.01, 91, 600)
        self.pitch_tier = call(manipulation, "Extract pitch tier")
        return self.pitch_tier
        
    def stylize_pitch(self,):
        if self.pitch_tier is not None:
            call(self.pitch_tier, "Stylize...",2.0,"semitones")
            tmp_pitch_point = self.pitch_point
            tmp_time_point = self.time_point
            self.set_time_and_pitch_point()
            if len(self.pitch_point) == 0:
                self.pitch_point = tmp_pitch_point
                self.time_point = tmp_time_point 
        else:
            print("pitch_tier is None")
            return 
        #self.pitch_tier.save("./Pitch_Tier/tmp.PitchTier")
        
    def set_total_duration(self,):
        # 정규 표현식을 사용하여 Total duration 값을 추출

        total_duration_match = re.search(r'Total duration: (\d+(\.\d+)?) seconds', str(self.pitch_tier))
        # Total duration 값이 존재하는지 확인 후 출력
        if total_duration_match:
            self.total_duration = float(total_duration_match.group(1))
        else:
            print(str(self.pitch_tier))
            print("Total duration not found.")
            
    def set_time_and_pitch_point(self,):
        self.pitch_tier.save("./tmp.PitchTier")#("./Pitch_Tier/tmp.PitchTier")
        r_file = open("./tmp.PitchTier")#("./Pitch_Tier/tmp.PitchTier",'r')
        
        self.pitch_point = []
        self.time_point = []
        while True:
            line = r_file.readline()
            if not line:
                break

            if 'number' in line:
                value = re.sub(r'[^0-9^.]', '', line)
                # 불필요한 . 제거 (뒤쪽 . 하나 제거)
                if value.count('.') > 1:
                    parts = value.split('.')
                    value = parts[0] + ''.join(parts[1:])
                if value != '':
                    self.time_point.append(round(float(value), 4))
            elif 'value' in line:
                value = re.sub(r'[^0-9^.]', '', line)
                # 불필요한 . 제거 (뒤쪽 . 하나 제거)
                if value.count('.') > 1:
                    parts = value.split('.')
                    value = parts[0] + ''.join(parts[1:])
                if value != '':
                    self.pitch_point.append(round(float(value), 4))
                
        if len(self.pitch_point)==0:
            print("왜 ????????????? 말이 돼??????????")
            while True:
                line = r_file.readline()
                if not line:
                    break
                print(line)
        r_file.close()
    
    #피치 포인트들 간의 거리        
    def extract_pd(self,):
        for i in range(len(self.pitch_point) - 1):
            point1 = (self.time_point[i], self.pitch_point[i])
            point2 = (self.time_point[i+1], self.pitch_point[i+1])
            #euclidean_distance
            distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            self.pd.append(round(distance,4))

    #피치 포인트 이동시간
    def extract_pt(self,):
        for i in range(len(self.time_point)-1):
            gap = self.time_point[i+1]-self.time_point[i]
            percent = round((gap/self.total_duration),4)
            self.pt.append(percent)
    
    #피치 포인트들 간의 변위
    def extract_pr(self,):
        if len(self.pitch_point) == 0:
            return
        total = max(self.pitch_point) - min(self.pitch_point)
        for i in range(1, len(self.pitch_point)):
            delta_y = abs(self.pitch_point[i] - self.pitch_point[i-1])
            
            r = round(delta_y / total,4)
            self.pr.append(r) 
    
    #피치 포인트들 간의 기울기
    def extract_ps(self,):
        for i in range(1, len(self.time_point)):
            delta_x = self.time_point[i] - self.time_point[i-1]
            delta_y = self.pitch_point[i] - self.pitch_point[i-1]
            
            if delta_x == 0:
                # 기울기를 정의할 수 없는 경우 (분모가 0인 경우)는 None을 추가
                print("이게 있으면 안되는뎀.")
                self.ps.append(None)
            else:
                slope = delta_y / delta_x
                self.ps.append(round(slope,4))
    def get_features(self,):
        
        self.pitch_tiers = self.get_pitch_tiers()
        self.set_time_and_pitch_point() # 추가됨 너무 짧은 문장들 (피치 티어 수가 원래 적은 문장) 때문에.
        self.stylize_pitch()
        #self.set_time_and_pitch_point()
        self.set_total_duration()
        feature = []
        #if self.args.features == 'intonation_with_text':
        self.extract_pd()
        self.extract_pt()
        self.extract_ps()
        self.extract_pr()
        
        #feature = self.pd + self.pt + self.pr + self.ps
        for t,d,s,r in zip(self.pt, self.pd, self.ps,self.pr):
#            tdsr = t * d * s * r
#            if tdsr < 0:
#                f = np.sqrt(abs(tdsr))
#                f = round(-1 * f,2)
#            else:
#                f = np.sqrt(abs(tdsr))
#                f = round(f, 2)
#            feature.append(f)
            td, rs =  t * d , r * s
            if s < 0:
                rs = np.sqrt(abs(rs))
                rs = round(-1 * rs, 2)
            td = np.sqrt(td)
            feature.append(td)#(t)
            feature.append(rs)#d)
            #feature.append(s)
            #feature.append(r)
        
        return feature
    def get_pitchs(self,):
        self.pitch_tiers = self.get_pitch_tiers()
        self.set_time_and_pitch_point() # 추가됨 너무 짧은 문장들 (피치 티어 수가 원래 적은 문장) 때문에.
        return self.pitch_point

    def get_intensity(self,):
        intensity = call(self.sound, "To Intensity...", 100.0, 0.0)
        return intensity.values[0]



def z_score_normalization(pitch_values):
    mean = np.mean(pitch_values)
    std = np.std(pitch_values)
    normalized_pitch = (pitch_values - mean) / std
    return normalized_pitch   
    

class data_loader(Dataset):
    def __init__(self, curr_data):
        self.task = 1  # 0은 normal, 1은 normalized_pitch
        self.embedding = 'linear' # cnn or linear 
        
        self.audio_model_name = "kresnik/wav2vec2-large-xlsr-korean"# "facebook/wav2vec2-base-960h"
        #self.audio_model = Wav2Vec2Model.from_pretrained("/home/ryumj/baseline-robust")#("facebook/wav2vec2-base-960h")
        self.hidden_size = 1024
        self.processor = Wav2Vec2Processor.from_pretrained(self.audio_model_name)

        self.data = []

        
        id = list(curr_data['id'])
        sentence = list(curr_data['sentence'])
        V = list(curr_data['V'])
        A = list(curr_data['A'])
        emotion = list(curr_data['emotion'])
        speaker = list(curr_data['speaker'])

        if self.task == 0:
            for i in tqdm(range((len(curr_data))),mininterval=10):
                # /home/ryumj/cupcake_backup/multimodal_dataset/dataset/clip_1_0.wav
                wav_path = '/home/ryumj/cupcake_backup/multimodal_dataset/dataset/'+ id[i]
                sr = 16000
                desired_duration = 5.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                
                self.data.append([V[i],A[i],waveform,re_sr])
        elif self.task == 1:
            for i in tqdm(range((len(curr_data))),mininterval=10):
                # /home/ryumj/cupcake_backup/multimodal_dataset/dataset/clip_1_0.wav
                wav_path = '/home/ryumj/cupcake_backup/multimodal_dataset/dataset/'+ id[i]
                sr = 16000
                desired_duration = 5.0
                waveform, re_sr = self.load_wav(wav_path, desired_duration, sr)
                
                sound = parselmouth.Sound(wav_path)
                intonation = Intonation_features(sound)
                pitch_tiers = intonation.get_pitchs()
                
                if not pitch_tiers:
                    print("피치 리스트가 비어있다고라")
                    continue
                
                #normalized_pitch = z_score_normalization(pitch_tiers)
                normalized_pitch = pitch_tiers
                
                if self.embedding == 'linear':
                    linear = torch.nn.Linear(len(normalized_pitch), self.hidden_size)
                    pitch_features = linear(torch.tensor(normalized_pitch, dtype=torch.float32))
                    print("pf :", pitch_features.shape)
                
                elif self.embedding == 'cnn':
                    # cnn embedding 
                    #print("nor", normalized_pitch)
                    pitch_features = torch.tensor(normalized_pitch).unsqueeze(-1)
                    #print("pf :", pitch_features.shape)
                    #pitch_features = pitch_features.mean(axis=0)
                
                self.data.append([V[i],A[i],pitch_features, waveform,re_sr])       

        print("데이터의 최종 길이 : ", len(self.data))
        print(self.data[:5])


    def __len__(self):
        return len(self.data)

    def load_wav(self,wav_path,desired_duration=10.0,resample_sr=16000):
        try:
            waveform, sample_rate = torchaudio.load(wav_path)
        except Exception as e:
            print(f"An error occurred at wav_path = {wav_path}: {str(e)}")
            return None, None
        
       # Calculate the duration of the audio
        audio_duration = waveform.size(1) / sample_rate

        # Check if the audio duration is longer than the desired duration
        if audio_duration > desired_duration:
            #Calculate the number of samples corresponding to 3 seconds
            desired_samples = int(desired_duration * sample_rate)

            # Get the last 5 seconds of the waveform
            waveform = waveform[:, -desired_samples:]
        else:
            # If the audio is shorter than 3 seconds, use it as is
            waveform = waveform
        #print("waveform type :", type(waveform))
        try:
            waveform = librosa.resample(y=np.array(waveform), orig_sr=sample_rate, target_sr=resample_sr)
        except Exception as e:
            print(f"An error occurred during resampling: {str(e)}")
            return None, None
        #waveform = librosa.resample(y=np.array(waveform), orig_sr=sample_rate, target_sr=resample_sr)

        #print('waveform 사이즈', torch.tensor(waveform).shape)
        return waveform, resample_sr


    def __getitem__(self,idx):
        return self.data[idx]
    
    def collate_fn(self, batch):

        #입력 : 배치 단위 데이터
        #출력 : torch.tensor(batch_V), torch.tensor(batch_A), torch.tensor(batch_D), audio_input_values

        batch_audio,batch_sr, batch_vad, batch_pitch = [],[],[],[]
        if self.task == 0 :
            for i,row in enumerate(batch):
                V, A, waveform, sr  = row
                #batch_vad.append([V,A,D])
                batch_vad.append([V,A])
                batch_audio.append(waveform)
                batch_sr.append(sr)
        elif self.task == 1:
            for i, row in enumerate(batch):
                V, A, pitch, waveform, sr = row
                batch_vad.append([V,A]) 
                batch_audio.append(waveform)
                batch_sr.append(sr)
                batch_pitch.append(pitch)           
       #-- audio processing --#
        batch_audio = [audio.flatten() for audio in batch_audio]
        batch_audio = list(batch_audio)
        audio_input_values = self.processor(audio=batch_audio,padding=True, sampling_rate=16000, return_tensors="pt").input_values
        audio_input_values = torch.tensor(audio_input_values)

        # Check and adjust sequence length
        desired_length = 16000  # Set the desired length for your model input
        batch_size, seq_length = audio_input_values.size()

        if seq_length < desired_length:
            # Pad if necessary
            padding_size = desired_length - seq_length
            padding = torch.zeros((batch_size, padding_size), dtype=audio_input_values.dtype)
            audio_input_values = torch.cat((audio_input_values, padding), dim=1)
        #audio_input_values = torch.tensor(batch_audio)
        
        if self.task == 0:
            audio_inputs = audio_input_values
        elif self.task == 1:
            pitch_values = torch.stack(batch_pitch)
            audio_inputs = ( pitch_values, audio_input_values)

            #print("pitch :", pitch_values)
            #print(pitch_values.shape)
        return audio_inputs,torch.tensor(batch_sr), torch.tensor(batch_vad)
