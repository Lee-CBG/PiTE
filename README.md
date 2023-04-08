# PiTE: Official Tensorflow Implementation

We provide the official Tensorflow & Keras implementation of training our PiTE: TCR-epitope binding affinity prediction pipeline using Transformer-based Sequence Encoder.

<p align="center"><img width=60% alt="Overview" src="https://github.com/Lee-CBG/PiTE/blob/main/figures/pipeline.png"></p>

## Publication
<b>PiTE: TCR-epitope Binding Affinity Prediction Pipeline using Transformer-based Sequence Encoder </b> <br/>
[Pengfei Zhang](https://github.com/pzhang84)<sup>1,2</sup>, [Seojin Bang](http://seojinb.com/)<sup>2</sup>, [Heewook Lee](https://scai.engineering.asu.edu/faculty/computer-science-and-engineering/heewook-lee/)<sup>1,2</sup><br/>
<sup>1 </sup>School of Computing and Augmented Intelligence, Arizona State University, <sup>2 </sup>Biodesign Institute, Arizona State University <br/>
Published in: **Pacific Symposium on Biocomputing (PSB), 2022.**


[Paper](https://www.worldscientific.com/doi/pdf/10.1142/9789811270611_0032) | [Code](https://github.com/Lee-CBG/PiTE) | [Poster](https://github.com/Lee-CBG/PiTE/blob/main/Supplementary/PiTE_PSB23_Poster.pdf) | [Slides](#) | Presentation ([YouTube](#))

## Dependencies

+ Linux
+ Python 3.6.13
+ Keras 2.6.0
+ TensorFlow 2.6.0

## Usage of PiTE

### 1. Clone the repository
```bash
git clone https://github.com/Lee-CBG/PiTE
cd PiTE/
pip install -r requirements.txt
```

### 2. Download training and testing dataset
- Data for baseline model can be downloaded [here](https://www.dropbox.com/sh/nnsgagtd7uf2a3f/AAB3-u3Is2l0YINfNQOg7kOYa?dl=0). The size is 5.16 GB.
- Data for other models such as Transformer, BiLSTM, and CNNs can be download [here](https://www.dropbox.com/sh/34t3qtivzn7z6u7/AACA05wYDUjDrej4qDcjnv_6a?dl=0). The size is 68.93 GB. Preprocess this data using preprocessing.ipynb file before intiating the training. 


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
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
