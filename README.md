# X-Ray Report Generation with CheXNet and GPT-2

This project generates radiology reports from chest X-ray images using a combination of CheXNet for image encoding and GPT-2 for text generation.

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.10 or later installed.
2. **CUDA Support**: If you plan to use GPU acceleration, ensure CUDA is installed and compatible with PyTorch.

## Steps

1. **Install requirements** `pip install -r requirements.txt`

2. **Data download** run data_download.py which downloads the Indiana University Chest X-Ray collection from Kaggle ([here](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university/data)) 

3. **Preprocess data** run preprocess_data.py - this extracts the relevant fields, aggregates images per study and cleans text data to replace 'XXXX' with a new special token, [REDACTED].

4. **Create dataset splits** run create_datasets.py to transform the images and save the dataset in train, test and validation splits.

5. **Prepare the GPT2 model and tokenizer** run add_special_token.py - this step is to add the [PAD] and [REDACTED] special tokens to the dataset