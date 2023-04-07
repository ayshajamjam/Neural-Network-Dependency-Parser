from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State
from operator import itemgetter

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0) 

        # print(self.output_labels)

        while state.buffer: 
            # TODO: Write the body of this loop for part 4 
            features = self.extractor.get_input_representation(words, pos, state)
            features_reshaped = features.reshape(1,-1)
            actions = self.model.predict(features_reshaped)[0]

            actions_list = []

            for a in range(len(actions)):   #91
                if(actions[a] > 0):
                    actions_list.append((self.output_labels[a], actions[a]))

            actions_list.sort(key=itemgetter(1),reverse=True)

            for a in actions_list:
                # Shifting the only word out of the buffer is also illegal unless the stack is empty.
                if(a[0][0] == 'shift'):
                    if(features[3] != 4 and features[4] == 4 and features[5] == 4 and features[0] != 4):
                        continue
                    else:   #shift
                        state.stack.append(state.buffer[-1])
                        state.buffer.pop()
                        # print(state.buffer)
                        # print(state.stack)
                        break
                elif(a[0][0] == 'left_arc'):
                    # arc-left not permitted when the stack is empty.
                    # The root node must never be the target of a left-arc 
                    if(features[0] == 3 or (features[0] == 4 and features[1] == 4 and features[2] == 4)):
                        continue
                    else:   # left_arc
                        child = state.stack.pop()
                        parent = state.buffer[-1]
                        state.deps.add((parent, child, a[0][1]))
                        break
                elif(a[0][0] == 'right_arc'):
                    # arc-right not permitted when the stack is empty.
                    if(features[0] == 4 and features[1] == 4 and features[2] == 4):
                        continue
                    else:   # right_arc
                        child = state.buffer[-1]
                        state.buffer[-1] = state.stack[-1]
                        parent = state.stack.pop()
                        state.deps.add((parent, child, a[0][1]))
                        break

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
