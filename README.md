# Neural-Network-Dependency-Parser2

**Background for project:** Chen, D., & Manning, C. (2014). [A fast and accurate dependency parser using neural networks.](https://www.emnlp2014.org/papers/pdf/EMNLP2014082.pdf)

This project was completed for my NLP class (Professor Bauer) at Columbia University.

## Task

**Goal:** Train a feed-forward neural network to predict the transitions of an arc-standard dependency parser.

**Input:** Representation of the current state (including words on the stack and buffer). 

**Output:** Transition (shift, left_arc, right_arc), together with a dependency relation label.

**Action Steps:** 
- Implement input representation for the neural net
- Decode the output of the network
- Specify the network architecture
- Train the model. 

## Packages, Tools, Libraries

- **Keras** (used to construct the neural network): a high-level Python API that allows you to easily construct, train, and apply neural networks. It is not a neural network library itself and requires TensorFlow as its backend. 
- **TensorFlow**: a high-level Python API that allows you to easily construct, train, and apply neural networks.

## Setup

1. pip install --upgrade setuptools wheel
2. python -m pip install tensorflow (CPU only setup)

## Files

- **conll_reader.py** - contains classes for representing dependency trees and reading in a CoNLL-X formatted data files. It is important that you understand how these data structures work, because you will use them to extract features below and also create dependency trees as parser output.


The class **DependencyEdge** represents a singe word and its incoming dependency edge. It includes the attribute variables id, word, pos, head, deprel. Id is just the position of the word in the sentence. Word is the word form and pos is the part of speech. Head is the id of the parent word in the tree. Deprel is the dependency label on the edge pointing to this label. Note that the information in this class is a subset of what is represented in the CoNLL format. 

The class **DependencyStructure** represents a complete dependency parse. The attribute deprels is a dictionary that maps integer word ids to DependencyEdge instances. The attribute root contains the integer id of the root note. 

The method **print_conll** returns a string representation for the dependency structure formatted in CoNLL format (including line breaks). 


- **get_vocab.py** - extract a set of words and POS tags that appear in the training data. This is necessary to format the input to the neural net (the dimensionality of the input vectors depends on the number of words). 
- **extract_training_data.py** - extracts two numpy matrices representing input-output pairs (as described below). You will have to modify this file to change the input representation to the neural network. 
- **train_model.py** - specify and train the neural network model. This script writes a file containing the model architecture and trained weights. 
- **decoder.py** - uses the trained model file to parse some input. For simplicity, the input is a CoNLL-X formatted file, but the dependency structure in the file is ignored. Prints the parser output for each sentence in CoNLL-X format. 
- **evaluate.py** - this works like decoder.py, but instead of ignoring the input dependencies it uses them to compare the parser output. Prints evaluation results. 

## Data

Use the standard split of the WSJ part of the Penn Treebank. The original Penn Treebank contains constituency parses, but these were converted automatically to dependencies. 

- data/train.conll - Training data. ~40k sentences
- data/dev.conll - Development data.  ~5k sentences. Use this to experiment and tune parameters. 
- data/sec0.conll - section 0 of the Penn Treebank. ~2k sentence -- good for quick initial testing.
- data/test.conll - Test data. ~2.5k sentences. Don't touch this data until you are ready for a final test of your parser.

## Dependency Format

The files are annotated using a modified CoNLL-X  format (CoNLL is the conference on Computational Natural Language learning -- this format was first used for shared tasks at this conference). Each sentences corresponds to a number of lines, one per word. Sentences are separated with a blank line. You will need to be able to read these annotations and draw dependency trees (by hand) in order to debug your parser.
Each line contains fields, seperated by a single tab symbol. The fields are, in order, as follows: 

- word ID (starting at 1)
- word form
- lemma
- universal POS tag
- corpus-specific POS tag (for our purposes the two POS annotations are always the same)
- features (unused)
- word ID of the parent word ("head"). 0 if the word is the root of the dependency tree. 
- dependency relation between the parent word and this word. 
- deps (unused)
- misc annotations (unused)

Any field that contains no entry is replaced with a _.

For example, consider the following sentence annotation: 

`1 The _ DT DT _ 2 dt _ _
2 cat _ NN NN _ 3 nsubj _ _
3 eats _ VB VB _ 0 root _ _
4 tasty _ JJ JJ _ 5 amod _ _
5 fish _ NN NN _ 3 dobj _ _
6 . _ . . _ 3 punct _ _`

The annotation corresponds to the following dependency tree:

![Example CoNLL annotation](conll_example.png)

## Part 1: Obtain the Vocabulary

Because we will use one-hot representations for words and POS tags, we will need to know which words appear in the data, and we will need a mapping from words to indices. 

Run the following:

`$python get_vocab.py data/train.conll data/words.vocab data/pos.vocab`

to generate an index of words and POS indices. This contains all words that appear more than once in the training data. The words file will look like this: 

`
<CD> 0
<NNP> 1
<UNK> 2
<ROOT> 3
<NULL> 4
blocking 5
hurricane 6
ships 7
`

The first 5 entries are special symbols. <CD> stands for any number (anything tagged with the POS tag CD), <NNP> stands for any proper name (anything tagged with the POS tag NNP). <UNK> stands for unknown words (in the training data, any word that appears only once). <ROOT> is a special root symbol (the word associated with the word 0, which is initially placed on the stack of the dependency parser). <NULL> is used to pad context windows. 

## Part 2: Extracting Input/Output Matrices for Training

To train the neural network we first need to obtain a set of input/output training pairs. More specifically, each training example should be a pair (x,y), where x is a parser state and y is the transition the parser should make in that state.

- python3 extract_training_data.py data/sec0.conll data/input_train.npy data/target_train.npy
- python3 extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
- python3 extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy

## Part 3: Designing and Training the network

### Network Topology

Now that we have training data, we can build the actual neural net. In the file train_model.py, write the function build_model(word_types, pos_types, outputs). word_types is the number of possible words, pos_types is the number of possible POS, and outputs is the size of the output vector. 

Start by building a network as follows:

- One **Embedding layer**, the input_dimension should be the number possible words, the input_length is the number of words using this same embedding layer. This should be 6, because we use the 3 top-word on the stack and the 3 next words on the buffer. The output_dim of the embedding layer should be 32.
- A **Dense hidden layer** of 100 units using relu activation. (note that you want to Flatten the output of the embedding layer first).  
- A **Dense hidden layer** of 10 units using relu activation. 
- An **output layer** using softmax activation.

  
Finally, the method should prepare the model for training, in this case using categorical crossentropy as the loss and the Adam optimizer with a learning rate of 0.01.

`model.compile(keras.optimizers.Adam(lr=0.01), loss="categorical_crossentropy")`

### Training a Model

The main function of train_model.py will load in the input and output matrices and then train the network. We will train the network for 5 epochs with a batch_size of 100. Training will take a while on a CPU-only setup. 

Finally it saves the trained model in an output file. 

- python3 train_model.py data/input_dev.npy data/target_dev.npy data/model_dev.h5
- python3 train_model.py data/input_train.npy data/target_train.npy data/model.h5

## Part 4: Greedy Parsing Algorithm - Building and Evaluating the Parser

We will now use the trained model to construct a parser. In the file decoder.py, take a look at the class Parser. The class constructor takes the name of a keras model file, loads the model and stores it in the attribute model. It also uses the feature extractor from part 2. 

The method parse_sentence(self, words, pos) takes as parameters a list of words and POS tags in the input sentence. The method will return an instance of DependencyStructure. 

The function first creates a State instance in the initial state, i.e. only word 0 is on the stack, the buffer contains all input words (or rather, their indices) and the deps structure is empty. 

The algorithm is the standard transition-based algorithm. As long as the buffer is not empty, we use the feature extractor to obtain a representation of the current state. We then call model.predict(features) and retrieve a softmax actived vector of possible actions. 
In principle, we would only have to select the highest scoring transition and update the state accordingly.

Unfortunately, it is possible that the highest scoring transition is not possible. arc-left or arc-right are not permitted the stack is empty. Shifting the only word out of the buffer is also illegal, unless the stack is empty. Finally, the root node must never be the target of a left-arc.

Instead of selecting the highest-scoring action, select the highest scoring permitted transition. The easiest way to do this is to create a list of possible actions and sort it according to their output probability (make sure the largest probability comes first in the list). Then go through the list until you find a legal transition. 

Running the program like this should print CoNLL formatted parse trees for the sentences in the input (note that the input contains dependency structures, but these are ignored -- the output is generated by your parser). 

`python3 decoder.py data/model.h5 data/dev.conll`

To evaluate the parser, run the program evaluate.py, which will compare your parser output to the target dependency structures and compute labeled and unlabeled attachment accuracy.

`python3 evaluate.py data/model.h5 data/dev.conll`
