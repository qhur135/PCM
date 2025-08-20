import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config
import torch.nn as nn

# Define a simple linear classifier
class AudioClassifier(torch.nn.Module):
    def __init__(self, ):
        super(AudioClassifier, self).__init__()

        self.task = 1
        self.embedding = 'linear' # cnn or linear 
        # for audio
        self.audio_model = Wav2Vec2Model.from_pretrained("./wav2vec2-large-korean")#("/home/ryumj/baseline-robust")#("facebook/wav2vec2-base-960h")
        self.a_hidden_size = self.audio_model.config.hidden_size #32000 #length
        self.audio_model.feature_extractor._freeze_parameters()
        # classifier
        self.num_classes = 2#3  # Number of classes
        self.linear = torch.nn.Linear(self.a_hidden_size, self.num_classes)
        
        if self.task == 1:
            # 어텐션 레이어
            self.attention_layer = torch.nn.MultiheadAttention(embed_dim=self.a_hidden_size, num_heads=8)
        
            # Pitch embedding
            if self.embedding == 'linear':
                self.pitch_embedding = torch.nn.Linear(self.a_hidden_size, self.a_hidden_size)
            elif self.embedding == 'cnn':
                self.pitch_embedding = torch.nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # Conv1D
                nn.ReLU(),  # 활성화 함수
                nn.Conv1d(in_channels=32, out_channels=self.a_hidden_size, kernel_size=3, stride=1, padding=1),  # Conv1D
                nn.AdaptiveAvgPool1d(1)  # 시간축 평균 풀링
                )
                            
    def forward(self, pitch, audio_inputs):
        #print("task : ", self.task)
        #print("embedding : ", self.embedding)
        if self.task == 0:
            #print("audio_inptus : ", audio_inputs.shape)
            # -- audio features -- #
            hidden_states = self.audio_model(audio_inputs).last_hidden_state
            audio_output = hidden_states.mean(dim=1) #[:,-1,:]

            #audio_output = audio_inputs

            audio_output = audio_output.to(torch.float32)

            final_output = self.linear(audio_output)#(audio_output)
        elif self.task == 1:
            # -- pitchs -- #
            pitches = pitch
             
            # -- raw_audio -- #
            tf_audio = audio_inputs
            wav2vec_outputs = self.audio_model(tf_audio).last_hidden_state
            
            # Pitch embedding
            pitches = pitches.float()
            
            if self.embedding == 'linear':
                pitch_embeds = self.pitch_embedding(pitches)
                
                pitch_embeds = pitch_embeds.unsqueeze(1).expand(-1, wav2vec_outputs.size(1), -1)
            elif self.embedding == 'cnn':
                pitches = pitches.squeeze(-1).unsqueeze(1)   # (batch_size, channel = 1, seq_len)
                pitch_embeds = self.pitch_embedding(pitches)   # (batch_size, hidden_dim, 1)
                pitch_embeds = pitch_embeds.transpose(1, 2) # (batch_size, new_seq_len, hidden_dim)
                # Expand pitch embeddings to match wav2vec2.0 output sequence length
                pitch_embeds = pitch_embeds.expand(-1, wav2vec_outputs.size(1), -1)
            # Apply attention mechanism
            attention_output, _ = self.attention_layer(query=pitch_embeds, key=wav2vec_outputs, value=wav2vec_outputs)#(wav2vec_outputs, pitch_embeds, pitch_embeds)
            
            # Average pooling over the sequence length
            pooled_output = attention_output.mean(dim=1)
            
            # Pass through the linear layer
            final_output = self.linear(pooled_output)
        
        return final_output
