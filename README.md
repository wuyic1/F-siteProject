F-site Model: A Transformer-based Molecular Generative Model for Fluorinated Molecular Design
Package Installation
Creating a Conda Environment
Create a dedicated Conda environment for this project:
conda create -n f-site python=3.8

Installing OpenNMT
This project relies on OpenNMT, an open-source neural machine translation and language modeling framework. For details, see OpenNMT-py GitHub. To install OpenNMT, run:
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
python setup.py install

Installing Other Dependencies
Install additional dependencies required for the project:
conda install cython
conda install spacy
conda install -c conda-forge rapidfuzz
pip install rdkit
pip install pandas
pip install scikit-learn

Usage
Data Preprocessing

Initial SMILES Data PreprocessingPerform initial preprocessing on raw molecular SMILES data (CSV format) for data cleaning and standardization:
python ./script/Sanitization_smi.py


Secondary SMILES Data PreprocessingFurther preprocess the cleaned and standardized SMILES data to filter based on predefined physicochemical properties and remove interfering compounds:
python ./script/Filter_data.py


Extract Fluorinated SubstructuresPrepare training corpora by extracting data for various fluorinated substructures:
python ./script/Get_patt_F.py


Data Triplet SplittingSplit the fluorinated substructure data into triplets (base-marked-orig) to match the F-site model's treinamento format:
python ./script/Cut_save_F.py


Vocabulary GenerationBuild a vocabulary based on the prepared training data:
python ./script/Generate_vocab.py


TokenizationTokenize the training corpora based on the generated vocabulary before model training:
python ./script/Tokenize_smiles.py



Data Splitting
This project supports two data splitting methods for model training: Random Splitting and Scaffold Splitting. Run one of the following:

For Random Splitting:
python ./script/Random_datasplit.py


For Scaffold Splitting (based on Bemis-Murcko scaffold grouping):
python ./script/Scaffold_datasplit.py



Training the Model
Ensure OpenNMT-py is installed as described above.
Train the two components of the F-site model using custom Transformer configuration files TrfmconfigM1.yaml and TrfmconfigM2.yaml:
onmt_train -config ./TrfmconfigM1.yaml
onmt_train -config ./TrfmconfigM2.yaml

Optional: If training is interrupted or you need to resume training, use the following command, specifying the checkpoint file:
onmt_train -config ./TrfmconfigM1.yaml -train_from ./model-directory/model_step_100000.pt

Predictions
Using the trained F-site models (Model 1 and Model 2), perform fluorination site prediction (M1) and molecular structure reconstruction with fluorinated groups (M2) on test data (prepared base data):
onmt_translate -model ./model-directory/model_step_100000.pt -src ./output/Test-src.txt -output ./output/Test-pred.txt -gpu 0 -n_best 3 -beam_size 3 -verbose

