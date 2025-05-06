# GenAID

This is the repository for GenAID, a **Gen**eralisable accent identification (**AID**) model across speakers. The code is built upon [speechbrain v0.5.16](https://github.com/speechbrain/speechbrain/tree/v0.5.16) with the original documentation [here](./README_speechbrain.md), and the AID dataset construction and code implementation by [CommonAccent](https://github.com/JuanPZuluaga/accent-recog-slt2022).

## I. Environment Setup

```bash
git clone https://github.com/jzmzhong/GenAID.git
cd GenAID
conda create -n speechbrain python==3.10
conda activate speechbrain
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install --editable .
```

## II. Data Preparation

### Download Common Voice

1. Download the dataset, which is available [here](https://commonvoice.mozilla.org/en/datasets). The specific data version used by this repo is ```Common Voice Corpus 17.0```.

2. Unzip the dataset after download.

    ```tar -xvf cv-corpus-17.0-2024-03-15-en.tar.gz```

### Process into Common Accent

1. Filter out the speech utterances with accent labels and process them.

2. Filter out accents with sufficient data and split into training/validation/testing sets. Note that there are two validation/testing sets, one for *seen* speakers (these speakers have sufficient data in the training set) and the other for *unseen* speakers (these speakers do not overlap with any of the speakers in the training set).

3. Processed training/validation/testing sets are available at:

    ```./recipes/CommonAccent/CommonAccent-CV17-spk-resplit```.

## III. Training

### Modify the Paths in the Configuration File

1. Please ensure that the ```data_folder``` field is the Common Voice dataset directory, and the ```csv_prepared_folder``` field is the Common Accent processed training/validation/testing sets directory, e.g. ```./recipes/CommonAccent/CommonAccent-CV17-spk-resplit```.

2. Also set the ```output_folder``` field to be the directory where you want to store the trained checkpoints and the ```rir_folder``` filed to be the directory where you want to store the noise dataset, downloaded and used in training for more robuts accent identification.

### Run the Model Training Script

```bash
cd ./recipes/CommonAccent
python train_GenAID.py train_GenAID_v6.yaml
```

### Trained Model

1. A trained model (with partial files for inference) is available at: https://drive.google.com/file/d/1slGrpZSu5g-nF7R-QMCmtGcjN3kw7lQj/view?usp=sharing

2. Please download and unzip it into the following directory for inference/embeddings extraction.

```bash
./recipes/CommonAccent/GenAID_v7
```

## IV. Inference

### Modify the Paths in the Configuration File

1. Please set ```pretrained_path``` field to be the checkpoint directory you want to inference, and the ```output_folder``` field to be the directory where you want to store the inference results (confusion matrices).

### Run the Model Inference Script

```bash
cd ./recipes/CommonAccent
python inference_GenAID.py inference_GenAID_v6.yaml
```

### Inference Results

Inference results are available here:

```bash
./recipes/CommonAccent/results
```

## V. Embeddings Extraction

### Data Preparation

1. Please process the data to extract embeddings from and place the original data under ```data_folder```, and the csv file of all files information under ```csv_prepared_folder```. An example of the data processing scripts is at ```./recipes/CommonAccent/VCTK/data_prep_VCTK.py```, with processed results at ```./VCTK/all_file_paths.csv```.

### Modify the Paths in the Configuration File

1. Please process the data to extract embeddings from and place the original data under ```data_folder```, and the csv file of all files information under ```csv_prepared_folder```. An example of the data processing scripts is at ```./recipes/CommonAccent/VCTK/data_prep_VCTK.py```, with processed results at ```./VCTK/all_file_paths.csv```.

2. Please set ```pretrained_path``` field to be the checkpoint directory you want to use to extract embeddings, and the ```output_folder``` field to be the directory where you want to store the extracted embeddings.

### Run the Embedding Extraction Script

```bash
cd ./recipes/CommonAccent
python extract_embeddings_GenAID.py extract_embeddings_GenAID_v6.yaml
```

## VI. Reference

CommonAccent: [Paper](https://www.isca-archive.org/interspeech_2023/zuluagagomez23_interspeech.pdf), [Code](https://github.com/JuanPZuluaga/accent-recog-slt2022), [Model](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english)

## VI. Citing
Please cite GenAID (part of the [AccentBox](https://ieeexplore.ieee.org/abstract/document/10888332) paper) if you use it for your research or business.

```bibtex
@inproceedings{zhong2025accentbox,
    author = {Zhong, Jinzuomu and Richmond, Korin and Su, Zhiba and Sun, Siqi},
    title = {{AccentBox: Towards High-Fidelity Zero-Shot Accent Generation}},
    booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year = {2025},
    pages = {1-5},
    doi = {10.1109/ICASSP49660.2025.10888332}
}
 
```