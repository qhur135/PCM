# Pitch Contour Model (PCM) with Transformer Cross-Attention for Speech Emotion Recognition

## Abstract
This repository contains the source code for the Interspeech2025 paper [Pitch Contour Model (PCM) with Transformer Cross-Attention for Speech Emotion Recognition](https://www.isca-archive.org/interspeech_2025/ryu25_interspeech.pdf).  

Pitch is important for distinguishing emotional states through intonation. To incorporate pitch contour patterns into Speech Emotion Recognition (SER) task, we propose the Pitch
Contour Model (PCM), which integrates pitch features with Transformer-based speech representations. PCM processes pitch features via linear embedding and combines them with Wav2Vec 2.0 extracted features using cross-attention. Experimental results show that PCM enhances SER performance, achieving state-of-the-art (SOTA) Valence-Arousal-Dominance (V-A-D) scores with V: 0.627, avg: 0.571 in MSP-Podcast v1.11 and V: 0.646, A: 0.744, D: 0.557 in IEMOCAP datasets. We observe that the effect of z-score normalization on pitch varies across datasets, with lower pitch variability conditions benefiting more from raw pitch values. Furthermore, our study suggests how pretraining and finetuning language mismatches, between English and Korean, affect the choice between CNNbased and linear embeddings for pitch representation.  

Authors: Minji Ryu, Ji-Hyeon Hur, Sung Heuk Kim, Gahgene Gweon  

---

## Run

- We have uploaded the code we actually used so that the experiments can be run with the provided configs file. 
- The codebase includes exploratory paths and ablation logic; some parts are not fully streamlined.

### English Dataset (IEMOCAP, MSP-Podcast)
```bash
python local_path/iemocap_audio_train.py --config local_path/configs/audio.json
```

#### `audio.json` configuration

* **text_cap_path**: Path to IEMOCAP dataset
* **text_msp_path**: Path to MSP-Podcast dataset
* **load_model_path**: Path to the saved model
* **logger_path**: Path for logging
* **hidden_size**: Hidden size of the saved model

#### Experiment settings

##### IEMOCAP

* **PCM-le-norm**

  * task: `intonation_and_wav2vec2`
  * normalized: `ok`
  * dataset: `iemocap`
  * embedding: `linear`

* **PCM-le-noNorm**

  * task: `intonation_and_wav2vec2`
  * normalized: `nope`
  * dataset: `iemocap`
  * embedding: `linear`

* **PCM-cnn**

  * task: `intonation_and_wav2vec2`
  * normalized: `ok`
  * dataset: `iemocap`
  * embedding: `cnn`

#### MSP-Podcast

* **PCM-le-norm**

  * task: `msp_intonation_and_wav2vec2`
  * normalized: `ok`
  * dataset: `msp_podcast`
  * embedding: `linear`

* **PCM-le-noNorm**

  * task: `msp_intonation_and_wav2vec2`
  * normalized: `nope`
  * dataset: `msp_podcast`
  * embedding: `linear`

* **PCM-cnn**

  * task: `msp_intonation_and_wav2vec2`
  * normalized: `ok`
  * dataset: `msp_podcast`
  * embedding: `cnn`

(*Other parameters in the JSON file are not relevant for experiments.*)

---

### Korean Dataset (Korean MVD)

* **`audio_k_dataset.py`**
  Experiment settings are configured at the top of the `data_loader` class.

* **`audio_k_model.py`**
  Experiment settings are configured at the top of the `AudioClassifier` class:

  * `self.task = 0` → noNorm
  * `self.task = 1` → norm
  * `self.embedding = 'linear'` or `'cnn'`

* **`audio_k_train.py`**
  At line 134:

  ```python
  text_path = 'dataset path'
  ```

---

