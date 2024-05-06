# Tomato Leaf Disease Identification using CNN

A CNN model trained using [TensorFlow](https://www.tensorflow.org/) under the GPUs provided by [Kaggle](https://www.kaggle.com/)

The model weights and checkpoints are provided under the `saved_models` folder for inference and further fine tuning.

## Dataset

PlantVillage Dataset, hosted on [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

## Training process
Refer to the training.ipynb Jupyter notebook.

## API Inference
Use the script inside the `api` folder to start a FastAPI server for inference

You need to first install the required depedencies using:

```sh
pip install -r requirements.txt
```

Start the server:

```
python3 main.py
```

## Client

Use the `index.html` inside the `client` folder to run the web client for making inference calls.

## References

Attached Research Paper has been used for references.
