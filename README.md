<h1 align="center"><span style="font-weight:normal">Diagnosis of depression based on facial multimodal data</h1>

<div align="center">
  
[Introduction](#intro) |
[Data Preparation](#preparation) |
</div>
<!-- 
<div align="center">  -->

## <a name="intro"></a>Introduction
*Depression is a serious mental health disease. Traditional scale-based depression diagnosis methods often have problems of strong subjectivity and high misdiagnosis rate, so it is particularly important to develop automatic diagnostic tools based on objective indicators. This study proposes a deep learning method that fuses multimodal data to automatically diagnose depression using facial video and audio data. We use spatiotemporal attention module to enhance the extraction of visual features and combine the Graph Convolutional Network (GCN) and the Long and Short Term Memory (LSTM) to analyze the audio features. Through the multi-modal feature fusion, the model can effectively capture different feature patterns related to depression. We conduct extensive experiments on the publicly available clinical dataset, the Extended Distress Analysis Interview Corpus (E-DAIC).*

## <a name="preparation"></a> Data

### Downloading the datasets

- For E-DAIC, the features are only available upon request [here](https://dcapswoz.ict.usc.edu/).

### Extracting non-verbal modalities

<details>
<summary> Click here for detailed tutorial </summary>

#### E-DAIC
- To pre-process the DAIC-WOZ features:

```
python3 -m venv venv
source venv/bin/activate
bash ./scripts/feature_extraction/extract-edaic-features.sh
deactivate
```

</details>

### Implementation Detail

Once all the data has been pre-processed, you should indicate the absule path to the directory where it is stored
in the 'configs/env_config.yaml' file for each one of the corresponding datasets
