<p align="center"><a href="https://novoic.com"><img src="https://assets.novoic.com/logo_320px.png" alt="Novoic logo" width="160"/></a></p>

# Novoic data engineering challenge

Thank you for your interest in Novoic. 

This take-home exercise is a way for us to evaluate some of the hands-on skills which we believe are important for success in this role. It should also give you an idea of the kinds of issues that we face on a daily basis, and give you a snapshot of the tasks that you might be working on at Novoic. This open-ended test should take you roughly 3 hours to complete, but feel free to use more or less. 

**Please read the whole exercise before starting.**

## The exercise
You will find at [this link](https://drive.google.com/file/d/10okw1LKGpzApm3Ecq0rFdlMkdebMOuFY/view?usp=sharing) a dataset containing a text folder, an audio folder and a `metadata.csv` file. Each text file (in `.xml` format) corresponds to exactly one audio file (in `.wav` format). This is a transcription of what is being said in the audio file. For privacy reasons, we have replaced the real audio files with silent audio files. The transcriptions have also been invented.

Each transcription is a conversation between an investigator and a patient (different patient in each transcription) – the patient ID can be found in the transcription. You can find patient level information in the metadata file, using the patient ID column. **You should assume that the transcript contains speech from both the patient and the investigator.**

Your task is to build a pipeline to get this data ready for training of machine learning models. Specifically, we want to build a pipeline which allows us to first filter the data based on the `metadata.csv` file, and then transform the selected `.xml` transcriptions into **fixed length numpy arrays of features**, and the selected `.wav` files into **fixed length numpy arrays of features**.

This pipeline should be able to deal with large amounts of data (1,500 hours of speech) and hence should be centered around a framework (we recommend using PySpark, but feel free to use any other framework) for **distributed data processing**.

Here are some machine learning tasks that your pipeline must be able to create data for:
- Classification of AD (Alzheimer's disease; `status = 1`) vs HC (healthy control, i.e. `status = 0`).
- Classification of AD-MCI (mild cognitive impairment due to Alzheimer's disease; `status = 1 AND mmse > 26`) vs HC.
- Classification of AD vs HC for patients in a certain age range. For example classification of AD vs HC for patients who are less than 60 years old (a loose proxy for preclinical Alzheimer’s).

We have provided two functions in `utils.py` in this repository. One which transforms text into a fixed length numpy array of features, and one which transforms audio (in the form of a 1D numpy array) into a fixed length numpy array of features. Feel free to use these functions. We ask that you showcase your exceptional software development skills in this mini-project, and that you **use Python**, keeping best practices (including PEP8 and OOP) in mind. If time permits, we would love to see some unit tests.

In order to run the functions in `utils.py` you will need to first run:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

To get started, please clone this repository and push it to GitHub as a private repository. The deliverable for this take-home exercise is this repo which you can share with us at jack@novoic.com. Please document your code and include a comprehensive `README.md`. Additionally, in the `README.md`, please include answers to the following questions (including code and/or diagrams if you wish):
- How would you deploy this repository on a Kubernetes cluster?
- Assume we now are using this repository as part of a product that we have deployed. How would you ensure that we can stream the data preprocessing? What technologies would you use? 

Finally, please zip your preprocessed dataset and include it in the repo for convenience. Alternatively, feel free to upload it to Google Drive or S3 and share it separately with jack@novoic.com. You are free to pick the output dataset format which you believe to be best suited for downstream applications.

Best of luck!
