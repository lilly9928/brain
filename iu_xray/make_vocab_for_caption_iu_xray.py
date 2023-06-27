import os
import numpy as np
import re
import pandas as pd

def make_vocab(input_dir,output_dir):
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    caption_length = []
    captions = pd.read_excel(input_dir, usecols=[2])
    set_caption_length = [None] * len(captions)

    for i in range(len(captions['caption'])):
        word_len = len(captions['caption'].values[i].split('\n'))
        sentence = captions['caption'].values[i].split('\n')
        for j in range(word_len):
            words = SENTENCE_SPLIT_REGEX.split(sentence[j])
            words = [w.strip() for w in words if len(w.strip()) > 0]
            vocab_set.update(words)
            set_caption_length[i] = len(words)
        caption_length += set_caption_length

    print(vocab_set)

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')

    with open(output_dir+'/vocab_caption.txt', 'w') as f:
        f.writelines([w + '\n' for w in vocab_list])

    print('Make vocabulary for caption')
    print('The number of total words of caption: %d' % len(vocab_set))



if __name__ == '__main__':
    input_dir = './iu_xray_data.xlsx'
    output_dir ='.'
    make_vocab(input_dir,output_dir)

#The number of total words of caption: 794