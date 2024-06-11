# VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild

## QuickStart Colab

:star: To try out speech editing or TTS Inference with VoiceCraft, the simplest way is using Google Colab.
Instructions to run are on the Colab itself.

1. To try [Speech Editing](https://colab.research.google.com/drive/1FV7EC36dl8UioePY1xXijXTMl7X47kR_?usp=sharing)
2. To try [TTS Inference](https://colab.research.google.com/drive/1lch_6it5-JpXgAQlUTRRI2z2_rk5K67Z?usp=sharing)

## QuickStart Command Line

:star: To use it as a standalone script, check out tts_demo.py and speech_editing_demo.py.
Be sure to first [setup your environment](#environment-setup).
Without arguments, they will run the standard demo arguments used as an example elsewhere
in this repository. You can use the command line arguments to specify unique input audios,
target transcripts, and inference hyperparameters. Run the help command for more information:
`python3 tts_demo.py -h`

## QuickStart Docker
:star: To try out TTS inference with VoiceCraft, you can also use docker. Thank [@ubergarm](https://github.com/ubergarm) and [@jayc88](https://github.com/jay-c88) for making this happen.

Tested on Linux and Windows and should work with any host with docker installed.
```bash
# 1. clone the repo on in a directory on a drive with plenty of free space
git clone git@github.com:jasonppy/VoiceCraft.git
cd VoiceCraft

# 2. assumes you have docker installed with nvidia container container-toolkit (windows has this built into the driver)
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.13.5/install-guide.html
# sudo apt-get install -y nvidia-container-toolkit-base || yay -Syu nvidia-container-toolkit || echo etc...

# 3. First build the docker image
docker build --tag "voicecraft" .

# 4. Try to start an existing container otherwise create a new one passing in all GPUs
./start-jupyter.sh  # linux
start-jupyter.bat   # windows

# 5. now open a webpage on the host box to the URL shown at the bottom of:
docker logs jupyter

# 6. optionally look inside from another terminal
docker exec -it jupyter /bin/bash
export USER=(your_linux_username_used_above)
export HOME=/home/$USER
sudo apt-get update

# 7. confirm video card(s) are visible inside container
nvidia-smi

# 8. Now in browser, open inference_tts.ipynb and work through one cell at a time
echo GOOD LUCK
```

## Environment setup
```bash
conda create -n voicecraft python=3.9.16
conda activate voicecraft

pip install -e git+https://github.com/facebookresearch/audiocraft.git@c5157b5bf14bf83449c17ea1eeb66c19fb4bc7f0#egg=audiocraft
pip install xformers==0.0.22
pip install torchaudio==2.0.2 torch==2.0.1 # this assumes your system is compatible with CUDA 11.7, otherwise checkout https://pytorch.org/get-started/previous-versions/#v201
apt-get install ffmpeg # if you don't already have ffmpeg installed
apt-get install espeak-ng # backend for the phonemizer installed below
pip install tensorboard==2.16.2
pip install phonemizer==3.2.1
pip install datasets==2.16.0
pip install torchmetrics==0.11.1
pip install huggingface_hub==0.22.2
# install MFA for getting forced-alignment, this could take a few minutes
conda install -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068
# install MFA english dictionary and model
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
# pip install huggingface_hub
# conda install pocl # above gives an warning for installing pocl, not sure if really need this

# to run ipynb
conda install -n voicecraft ipykernel --no-deps --force-reinstall
```

If you have encountered version issues when running things, checkout [environment.yml](./environment.yml) for exact matching.

## Inference Examples
Checkout [`inference_speech_editing.ipynb`](./inference_speech_editing.ipynb) and [`inference_tts.ipynb`](./inference_tts.ipynb)

## Gradio
### Run in colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IOjpglQyMTO2C3Y94LD9FY0Ocn-RJRg6?usp=sharing)

### Run locally
After environment setup install additional dependencies:
```bash
apt-get install -y espeak espeak-data libespeak1 libespeak-dev
apt-get install -y festival*
apt-get install -y build-essential
apt-get install -y flac libasound2-dev libsndfile1-dev vorbis-tools
apt-get install -y libxml2-dev libxslt-dev zlib1g-dev
pip install -r gradio_requirements.txt
```

Run gradio server from terminal or [`gradio_app.ipynb`](./gradio_app.ipynb):
```bash
python gradio_app.py
```
It is ready to use on [default url](http://127.0.0.1:7860).

### How to use it
1. (optionally) Select models
2. Load models
3. Transcribe
4. (optionally) Tweak some parameters
5. Run
6. (optionally) Rerun part-by-part in Long TTS mode

### Some features
Smart transcript: write only what you want to generate

TTS mode: Zero-shot TTS

Edit mode: Speech editing

Long TTS mode: Easy TTS on long texts


## Training
To train an VoiceCraft model, you need to prepare the following parts:
1. utterances and their transcripts
2. encode the utterances into codes using e.g. Encodec
3. convert transcripts into phoneme sequence, and a phoneme set (we named it vocab.txt)
4. manifest (i.e. metadata)

Step 1,2,3 are handled in [./data/phonemize_encodec_encode_hf.py](./data/phonemize_encodec_encode_hf.py), where
1. Gigaspeech is downloaded through HuggingFace. Note that you need to sign an agreement in order to download the dataset (it needs your auth token)
2. phoneme sequence and encodec codes are also extracted using the script.

An example run:

```bash
conda activate voicecraft
export CUDA_VISIBLE_DEVICES=0
cd ./data
python phonemize_encodec_encode_hf.py \
--dataset_size xs \
--download_to path/to/store_huggingface_downloads \
--save_dir path/to/store_extracted_codes_and_phonemes \
--encodec_model_path path/to/encodec_model \
--mega_batch_size 120 \
--batch_size 32 \
--max_len 30000
```
where encodec_model_path is avaliable [here](https://huggingface.co/pyp1/VoiceCraft). This model is trained on Gigaspeech XL, it has 56M parameters, 4 codebooks, each codebook has 2048 codes. Details are described in our [paper](https://jasonppy.github.io/assets/pdfs/VoiceCraft.pdf). If you encounter OOM during extraction, try decrease the batch_size and/or max_len.
The extracted codes, phonemes, and vocab.txt will be stored at `path/to/store_extracted_codes_and_phonemes/${dataset_size}/{encodec_16khz_4codebooks,phonemes,vocab.txt}`.

As for manifest, please download train.txt and validation.txt from [here](https://huggingface.co/datasets/pyp1/VoiceCraft_RealEdit/tree/main), and put them under `path/to/store_extracted_codes_and_phonemes/manifest/`. Please also download vocab.txt from [here](https://huggingface.co/datasets/pyp1/VoiceCraft_RealEdit/tree/main) if you want to use our pretrained VoiceCraft model (so that the phoneme-to-token matching is the same).

Now, you are good to start training!

```bash
conda activate voicecraft
cd ./z_scripts
bash e830M.sh
```

It's the same procedure to prepare your own custom dataset. Make sure that if

## Finetuning
You also need to do step 1-4 as Training, and I recommend to use AdamW for optimization if you finetune a pretrained model for better stability. checkout script `./z_scripts/e830M_ft.sh`.

If your dataset introduce new phonemes (which is very likely) that doesn't exist in the giga checkpoint, make sure you combine the original phonemes with the phoneme from your data when construction vocab. And you need to adjust `--text_vocab_size` and `--text_pad_token` so that the former is bigger than or equal to you vocab size, and the latter has the same value as `--text_vocab_size` (i.e. `--text_pad_token` is always the last token). Also since the text embedding are now of a different size, make sure you modify the weights loading part so that I won't crash (you could skip loading `text_embedding` or only load the existing part, and randomly initialize the new)

