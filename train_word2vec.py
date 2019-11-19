from utils import load_recipes, tok
from tqdm import tqdm
import os
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from utils import find_words
import argparse

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--data_dir', default='./data')
args = parser.parse_args()
data_dir = args.data_dir

print('Load documents...')
file_path = os.path.join(data_dir, 'recipesV1.json')
recipes = [x for x in load_recipes(file_path) if x['partition']=='train']

print('Tokenize...')
all_sentences = []
for entry in tqdm(recipes):
    all_sentences.append(find_words(entry['title']))
    insts = entry['instructions']
    sentences = [find_words(inst) for inst in insts]
    all_sentences.extend(sentences)
print('number of sentences =', len(all_sentences))

print('Train Word2Vec model...')
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0
    def on_epoch_begin(self, model):
        print('-' * 40)
        print("Epoch #{} start".format(self.epoch))
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        print('total_train_time = {:.2f} s'.format(model.total_train_time))
        self.epoch += 1

epoch_logger = EpochLogger()
model = Word2Vec(
    all_sentences, size=300, window=10, min_count=10, 
    workers=20, iter=10, callbacks=[epoch_logger])

vocab_inst = model.wv.index2word
print('vocab_inst size =', len(vocab_inst))
with open(os.path.join(data_dir, 'vocab_inst.txt'), 'w') as f:
    f.write('\n'.join(vocab_inst))

model.wv.save(os.path.join(data_dir, 'word2vec.bin'))
