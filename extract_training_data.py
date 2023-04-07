from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np
import tensorflow as tf

class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence: 
            self.buffer = list(reversed(sentence))
        self.deps = set() 
    
    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add( (self.buffer[-1], self.stack.pop(),label) )

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)

   

def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label) 
        elif rel == "right_arc":
            state.right_arc(label) 
         
    return state.deps
   
class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None    
    def __repr__(self):
        return "<ROOT>"

     
def get_training_instances(dep_structure):

    deprels = dep_structure.deprels
    
    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1
 
    seq = []
    while state.buffer: 
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy() 
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else: 
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq   


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']




class FeatureExtractor(object):
       
    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)        
        self.pos_vocab = self.read_vocab(pos_vocab_file)        
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))
    
        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file: 
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab     
    
    def get_input_representation(self, words, pos, state):
        # TODO: Write this method for Part 2

        # Isolate buff and stack words from state
        stack = state.stack
        buff = state.buffer

        # Create array [stack[-1], stack[-2], stack[-3], buff[-1], buff[-2], buff[-3]]
        # Contain words
        arr_stack = []
        arr_buff = []

        # Contain indexes
        arr_stack_index = []
        arr_buff_index = []

        # print(words)
        # print(pos)

        # Get stack indices
        for i, item in enumerate(reversed(stack)):
            if(i == 3):
                break
            arr_stack.append(item)
            arr_stack_index.append(item)

        """
        The first 5 entries are special symbols. 
        <CD> stands for any number (anything tagged with the POS tag CD), 
        <NNP> stands for any proper name (anything tagged with the POS tag NNP). 
        <UNK> stands for unknown words (in the training data, any word that appears only once). 
        <ROOT> is a special root symbol (the word associated with the word 0, which is initially placed on the stack of the dependency parser). 
        <NULL> is used to pad context windows. 
        """
        
        # Substitute words/tags in arr
        for i in range(len(arr_stack)):
            # Number --> CD
            if(pos[arr_stack[i]] == 'CD'):
                arr_stack[i] = '<CD> ' + words[arr_stack[i]]   # 0
                arr_stack_index[i] = 0
            # Proper name --> NNP
            elif(pos[arr_stack[i]] == 'NNP'):
                arr_stack[i] = '<NNP> ' + words[arr_stack[i]]  # 1
                arr_stack_index[i] = 1
            # Root symbol --> ROOT
            elif(arr_stack[i] == 0):
                arr_stack[i] = '<ROOT>' # 3
                arr_stack_index[i] = 3
            # Word seen only once --> UNK
            elif(words[arr_stack[i]].lower() not in self.word_vocab.keys()):
                arr_stack[i] = '<UNK> ' + words[arr_stack[i]]  # 2
                arr_stack_index[i] = 2
            # Word seen more than once
            elif(words[arr_stack[i]].lower() in self.word_vocab.keys()):
                word = words[arr_stack[i]].lower()
                arr_stack[i] = word
                arr_stack_index[i] = self.word_vocab[word]

        while(len(arr_stack) != 3):
            arr_stack.append('<NULL>')
            arr_stack_index.append(4)

        # print(arr_stack)
        # print(arr_stack_index)

        # Get buff indices
        for i, item in enumerate(reversed(buff)):
            if(i == 3):
                break
            arr_buff.append(item)
            arr_buff_index.append(item)

        # Substitute words/tags in arr
        for i in range(len(arr_buff)):
            # Number --> CD
            if(pos[arr_buff[i]] == 'CD'):
                arr_buff[i] = '<CD> ' + words[arr_buff[i]]   # 0
                arr_buff_index[i] = 0
            # Proper name --> NNP
            elif(pos[arr_buff[i]] == 'NNP'):
                arr_buff[i] = '<NNP> ' + words[arr_buff[i]]   # 1
                arr_buff_index[i] = 1
            # Root symbol --> ROOT
            elif(arr_buff[i] == 0):
                arr_buff[i] = '<ROOT>' # 3
                arr_buff_index[i] = 3
            # Word seen only once --> UNK
            elif(words[arr_buff[i]].lower() not in self.word_vocab.keys()):
                arr_buff[i] = '<UNK> ' + words[arr_buff[i]]  # 2
                arr_buff_index[i] = 2
            # Word seen more than once
            elif(words[arr_buff[i]].lower() in self.word_vocab.keys()):
                word = words[arr_buff[i]].lower()
                arr_buff[i] = word
                arr_buff_index[i] = self.word_vocab[word]

        while(len(arr_buff) != 3):
            arr_buff.append('<NULL>')
            arr_buff_index.append(4)

        # print(arr_buff)
        # print(arr_buff_index)

        arr_word = arr_stack + arr_buff
        arr_index = arr_stack_index + arr_buff_index

        # print(arr_word)
        # print(arr_index)
        # print('\n')

        return np.asarray(arr_index)

    def get_output_representation(self, output_pair):  
        # TODO: Write this method for Part 2

        arr = np.zeros(91)
        index = self.output_labels[output_pair]
        arr[index] = 1

        print(output_pair)
        print(arr)

        # matrix_transitions = tf.keras.utils.to_categorical(dep_relations, num_classes=91, dtype ="int32")

        return arr

     
    
def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0 
    for dtree in conll_reader(in_file): 
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)
       


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 


    with open(sys.argv[1],'r') as in_file:   

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        # np.save(sys.argv[2], inputs)
        # np.save(sys.argv[3], outputs)


