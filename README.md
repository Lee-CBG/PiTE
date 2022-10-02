# PiTE: Official Tensorflow Implementation

We provide the official Tensorflow & Keras implementation of training our PiTE: TCR-epitope binding affinity prediction pipeline using Transformer-based Sequence Encoder.

## Dependencies

+ Linux
+ Python 3.6.13
+ Keras 2.6.0
+ TensorFlow 2.6.0

## Usage of PiTE

### 1. Clone the repository
```bash
$ git clone https://github.com/Lee-CBG/PiTE
$ cd PiTE/
$ conda env create -n PiTE -f environment.yml
```

### 2. Download training and testing dataset
- Data for baseline model can be downloaded [here](https://drive.google.com/drive/folders/1bXGenR3e6GgAuiEnfiG4N2RTZb3cRaUX?usp=sharing). The size is 5.16 GB.
- Data for other models such as Transformer, BiLSTM, and CNNs can be download [here](https://drive.google.com/drive/folders/12jb8BshG9mJI6xXuRdQrClJJ-PgkWGgG?usp=sharing). The size is 68.93 GB. Preprocess this data using preprocessing.ipynb file before intiating the training. 


### 3. Train models
An example for training the transformer-based model

```python
python -W ignore main.py \
--nns transformer \
--split tcr \
--gpu 0 \
--run 0 \
--seed 42
```

## Citation
If you use this code or use our PiTE for your research, please cite our [paper]
