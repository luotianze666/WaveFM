# [NAACL 2025] WaveFM: A High-Fidelity and Efficient Vocoder Based on Flow Matching
Arxiv: https://arxiv.org/abs/2503.16689

Our audio demo page is available at https://luotianze666.github.io/WaveFM.

Our checkpoints are available in this repository.

## Basic Usage
## 1. Get Mel-Spectrograms
You can extract Mel-spectrograms for both the training and inference processes using the following command.
```python
python dataset.py -i your_audio_path -o your_mel_saving_path
```
## 2. Train WaveFM
First modify the paths for `training` in `src/params.py` as needed, and then use `python src/train.py` to train WaveFM.

If `trainCheckPointPath` exists, the training process will resume from the corresponding checkpoints. Otherwise, new checkpoints will be saved to this path. During the training process, checkpoints are saved every `trainCheckPointSavingStep` steps.

**Note:** A coefficient $\frac{1}{\min(0.1,1-t)}$ is applied to our mean square loss function, where $t$ is uniformly sampled from $[0,1]$. Therefore, $7.5 \times 10^{-5}$ is not an actually small initial learning rate. Increasing the learning rate (e.g., to $1 \times 10^{-4}$) might improve performance but might also introduce instability during the training process.
```python
# parameters for training
trainInitialLR = 7.5e-5,
trainFinalLR = 5e-6,
trainBetas = (0.9, 0.99),
trainWeightDecay = 5e-4,
trainSteps = 1000000,
trainBatch = 24,
trainCheckPointSavingStep = 10000,
trainAudiosPath = "Your Audios",
trainMelsPath = "Your Mels",
trainCheckPointPath = "./checkpoints/WaveFM_0",
trainGPUs = [0], # [] represents CPU
```
## 3. Use Checkpoints For Inference 
First modify the paths for `inference` in `src/params.py` as needed, and then use `python src/inference.py` to run inference for WaveFM.

**Note:** The inference time displayed on the tqdm bar doesn't include the file loading time. Additionally, the inference speed can be much faster for longer audios due to a higher degree of parallelism.
```python
# parameters for inference
inferenceSteps = 6,
inferenceMelsPath = "Your Mels For Inference",
inferenceSavingPath = "./generation",
inferenceCheckPointPath = "./checkpoints/WaveFM_1000000", 
                        # "./distilled_checkpoints/Distilled_WaveFM_25000",
inferenceWithGPU = True, # use cuda:0 or CPU
```
## 4. Distill WaveFM
First modify the paths for `distillation` in `src/params.py` as needed, and then use `python src/distillation.py` to distill WaveFM.

`distillDelta` is the time step length used in the distillation process, and `distillModelPath` specifies the saving path of the teacher model.

If `distillCheckPointPath` exists, the distillation process will resume from the corresponding checkpoints. Otherwise, new checkpoints will be saved to this path. During the distillation process, checkpoints are saved every `distillCheckPointSavingStep` steps.

**Note:** In our experiments, a distillation process of 25,000 steps is sufficient. Further increasing the distillation steps does not improve performance.

```python
# parameters for distillation
distillInitialLR = 2e-5,
distillFinalLR = 5e-6,
distillBetas = (0.8, 0.95),
distillWeightDecay = 1e-2,
distillDeltaT = 0.01,
distillSteps = 25000,
distillBatch = 24,
distillCheckPointSavingStep = 5000,
distillModelPath = "./checkpoints/WaveFM_1000000",
distillCheckPointPath = "./distilled_checkpoints/Distilled_WaveFM_0",
distillAudiosPath = "Your Audios",
distillMelsPath = "Your Mels",
distillGPUs = [0], # [] represents CPU
```

## Package Requirements

WaveFM has been tested on `python 3.9` with the following requirements, and it should also work fine with the latest version.

```python
torch           2.0.1+cu118
torchaudio      2.0.2
```

## BibTex
```
@inproceedings{luo2025wavefm,
  title={Wavefm: A high-fidelity and efficient vocoder based on flow matching},
  author={Luo, Tianze and Miao, Xingchen and Duan, Wenbo},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={2187--2198},
  year={2025}
}
```
