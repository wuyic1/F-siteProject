## F-site Model: A Transformer-based Molecular Generative Model for Fluorinated Molecular Design

This repository contains code and instructions for training and using the F-site model, which is a Transformer-based molecular fluorine modification model used to predict the fluorine substitution points of small molecule compounds.

### Package Installation

#### Create Conda Environment

```bash
conda create -n f-site python=3.8
```

#### Install OpenNMT-py

This project is based on [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), an open-source toolkit for neural machine translation and large language modeling.

```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
python setup.py install
```

#### Install Additional Dependencies

```bash
conda install cython
conda install spacy
conda install -c conda-forge rapidfuzz
pip install rdkit
pip install pandas
pip install scikit-learn
```

---

### Usage

#### Data Preprocessing

1. **Sanitize and Normalize SMILES**

   Clean and standardize the original SMILES data:

   ```bash
   python ./script/Sanitization_smi.py
   ```

2. **Filter Based on Properties**

   Apply filtering based on predefined physicochemical properties:

   ```bash
   python ./script/Filter_data.py
   ```

3. **Extract Fluorinated Patterns**

   Generate datasets containing fluorinated substructures:

   ```bash
   python ./script/Get_patt_F.py
   ```

4. **Convert to Training Triplets**

   Convert the data into triplet format: `base`, `marked`, and `original`:

   ```bash
   python ./script/Cut_save_F.py
   ```

5. **Generate Vocabulary**

   Create vocabulary for tokenization:

   ```bash
   python ./script/Generate_vocab.py
   ```

6. **Tokenize SMILES**

   Tokenize data using the vocabulary:

   ```bash
   python ./script/Tokenize_smiles.py
   ```

---

#### Data Splitting

Split the dataset for training using either random or scaffold-based methods:

```bash
python ./script/Random_datasplit.py            # Random splitting
# or
python ./script/Scaffold_datasplit.py          # Scaffold splitting (based on Bemis-Murcko scaffolds)
```

---

#### Training Model

Train the F-site model using Transformer configurations:

```bash
onmt_train -config ./TrfmconfigM1.yaml
onmt_train -config ./TrfmconfigM2.yaml
```

**Optional:** Resume training from a saved checkpoint:

```bash
onmt_train -config ./TrfmconfigM1.yaml -train_from ./model-directory/model_step_100000.pt
```

---

#### Predictions

Run predictions using the trained models:

```bash
onmt_translate -model ./model-directory/model_step_100000.pt \
               -src ./output/Test-src.txt \
               -output ./output/Test-pred.txt \
               -gpu 0 \
               -n_best 3 \
               -beam_size 3 \
               -verbose
```

This command performs site prediction (Model 1) and molecular reconstruction with fluorinated groups (Model 2), based on your test input.
