This is a pet project created to demonstrate machine learning skills in the field of deepfake detection. The project is currently under active development and not yet fully complete.

Topic
=====

Deepfakes are becoming a major concern with the growing popularity of AI-generated content. Detecting deepfakes is essential to prevent fraud, misinformation, and other potential threats.

This project explores Fourier-based analysis (FFT) as a method for detecting periodic artifacts often present in deepfake images. The core of the model is a CNN architecture trained on FFT maps extracted from facial images.

The implementation is inspired by the Silent-Face-Anti-Spoofing project (https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master), which was originally designed for face spoofing detection.


How to use:
===========

First of all, you need to prepare the data. For this purpose, you can use data_processing.py. All you need to do is specify the root folder with the dataset and the target folder where the images of a specific class are located, as well as indicate where the images will be collected and their size.

    python src/data_processing.py --root_folder ds_to_process/ --target_folder live/ --output_folder dataset/train/live/ --out_size 128

You can run train process when you are done with data preparation. Just configure `src/config.yaml` for this.

To run the train process, you need to use `src/run.train.py`. It is easy

    python src/run_train.py --config_path src/config.yaml

To test the model or models using test dataset, use `src/test.py`. It will be looking like this (you can test one specific model or all the models after training)

    python src/test.py --model_path models/2025_04_08_19_16/ --data_path_live dataset/test/live/ --data_path_fake dataset/test/fake/

The example of testing real model are listed below

`models/2025_04_07_21_16/MultiFTNet_model__12.pth
{'accuracy': 0.8893168271435121, 'precision': 0.88937452575616, 'recall': 0.8893168271435121, 'f1_score': 0.8893178491032441, 'average_precision': array([0.95421896, 0.96429587]), 'loss': 0.0}`

If you would like to tune some parameters, please, see `src/findParametersWithOptuna.py`

On the next step, you can conver PyTorch model to ONNX and OpenVINO. Just use `src/to_ovine.py` for this. Is is also super easy.

    python rc/to_ovine.py --model_path models/2025_04_07_21_16/MultiFTNet_model__12.pth --out_folder converted/

And the last spep is inference of the converted model in accordance with your bussiness logic.

    pytgon src/OVINEvisualization.py  --ovine_model converted/ovinemodel.xml --img test.jpg

and you will see something like this `[0.04510313 0.9548969 ]`, where the values are probabilities of the classes [real, spoof]