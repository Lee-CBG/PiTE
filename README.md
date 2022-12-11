# PiTE: Official Tensorflow Implementation

We provide the official Tensorflow & Keras implementation of training our PiTE: TCR-epitope binding affinity prediction pipeline using Transformer-based Sequence Encoder.

<p align="center"><img width=60% alt="Overview" src="https://github.com/Lee-CBG/PiTE/blob/main/figures/pipeline.pdf"></p>

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

```bash
python -W ignore main.py \
--nns transformer \
--split tcr \
--gpu 0 \
--run 0 \
--seed 42
```

## Citation
If you use this code or use our PiTE for your research, please cite our paper:
```
@inproceedings{zhang2022pite,
  title={PiTE: TCR-epitope Binding Affinity Prediction Pipeline using Transformer-based Sequence Encoder},
  author={Zhang, Pengfei and Bang, Seojin and Lee, Heewook},
  booktitle={PACIFIC SYMPOSIUM ON BIOCOMPUTING 2023: Kohala Coast, Hawaii, USA, 3--7 January 2023},
  pages={347--358},
  year={2022},
  organization={World Scientific}
}
```

## License

Released under the [ASU GitHub Project License](./LICENSE).
