# Neural Network Dependency Parser

## Task

Train a feed-forward neural network to predict the transitions of an arc-standard dependency parser. The input to this network will be a representation of the current state (including words on the stack and buffer). The output will be a transition (shift, left_arc, right_arc), together with a dependency relation label.

## Instructions

### Part 2
- python3 extract_training_data.py data/sec0.conll data/input_train.npy data/target_train.npy
- python3 extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
- python3 extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy

### Part 3
- python3 train_model.py data/input_dev.npy data/target_dev.npy data/model_dev.h5
- python3 train_model.py data/input_train.npy data/target_train.npy data/model.h5

### Part 4
- python3 decoder.py data/model.h5 data/dev.conll
- python3 evaluate.py data/model.h5 data/dev.conll


## Packages, Tools, Libraries

- **Keras**: a high-level Python API that allows you to easily construct, train, and apply neural networks. It is not a neural network library itself and requires TensorFlow as its backend.
- **TensorFlow**: a high-level Python API that allows you to easily construct, train, and apply neural networks.

## Setup Instructions

pip install --upgrade setuptools wheel
python3s -m pip install tensorflow
python3 get_vocab.py data/train.conll data/words.vocab data/pos.vocab