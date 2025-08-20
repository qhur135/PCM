# PCM
Code for our paper accepted at interspeech 2025 (camera ready submitted)


- This codebase includes a variety of experiments and configurations, not all of which are directly related to the final version of the paper.
- Some sections may contain hardcoded logic, commented-out blocks, or partial implementations used for ablation studies.
- A cleaned version of the code and detailed README are scheduled.

---

## Running (English version)

```bash
python local_path/iemocap_audio_train.py --config local_path/configs/audio.json
```

### `audio.json` configuration

* **text_cap_path**: Path to IEMOCAP dataset
* **text_msp_path**: Path to MSP-Podcast dataset
* **load_model_path**: Path to the saved model
* **logger_path**: Path for logging
* **hidden_size**: Hidden size of the saved model

### Experiment settings

#### IEMOCAP

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

### Korean version files

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

