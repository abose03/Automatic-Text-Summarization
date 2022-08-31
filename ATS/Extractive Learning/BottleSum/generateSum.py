from utils.bottleEx_extractive_funs import elimination_beam_search
import argparse
from tqdm import tqdm, trange
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd


# specify configuration.
config = {
    'S1_path': 'S1.txt',
    'S2_path': 'S1.txt',     
    'out_name': 'train_Summaries_100_600.txt', 
    'max_tokens_batch': 20000, 
    'start': 0, 
    'log_interval':  1,
    'window': 50, 
    'rem_words': 3, 
    'beam': 1, 
    'min_words': 1,     
    'lowercase': 'store_true',
    'device': 'cuda:0'
}

print(config)

# loading tokenizer and model.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(config['device'])

out_str = ''
count = 0
start = config['start']
 
start_time = time.time()

#df = pd.read_excel("Inference.xlsx")
#examples_S1 = df['prompt']
#examples_S2 = df['answer']

# reading all examples from s1 file (sentence 1)
with open(config['S1_path'], 'r') as f:
    examples_S1 = f.read().split("\n")
    
# reading all examples from s2 frile (sentence 2)
with open(config['S2_path'], 'r') as f:
    examples_S2 = f.read().split("\n")

print("Lengths:", len(examples_S1), len(examples_S2))
examples = list(zip(examples_S1, examples_S2))


# split large articel into small sentences and also 
# if the sentence is smaller than 10 words. then it 
# will add to previous sentence.
def split_to_sentences(text):
    
    l = text.split(".")
    ll = []
    for i in range(len(l)):
        if len(l[i].split()) < 10:
            if len(ll)>0:
                ll[-1] = ll[-1] + " " + l[i]
        else:
            ll.append(l[i])
            
    return ll

# dry run to check sentence split working correct or not.
lll = []
for i in range(len(examples_S1)):
    ll = split_to_sentences(examples_S1[i])
    lll.extend(ll)
len(lll)

# looping thourgh all examples.
out_str = ''
for example in tqdm(examples_S1, mininterval=2, desc='  - (Generating from examples)   ', leave=False):

    # continue until start.
    if count < start:
        count += 1
        continue
    
    
    # at log_interval, save intermediate output and output stats
    if count % config['log_interval'] == 0:
        with open(config['out_name'] , 'w') as f:
            f.write(out_str)
        c_time = time.time() - start_time
        print('count {} || time {} || {} s/it'.format(count, c_time, c_time/(count - start + 0.00001)))
    
    # split the arctile to sentences.
    ll = split_to_sentences(example)
        
    
    cur_summary = ""
    for i in trange(len(ll)-1):
        # get S1 and S2
        S1 = ll[i]
        S2 = ll[i+1]

        # summarize!
        result = elimination_beam_search(S1 = S1, S2 = S2,
                                k = config['beam'], 
                                rem_words = config['rem_words'], 
                                max_tokens_batch = config['max_tokens_batch'],
                                tok_method = 'moses',
                                autocap = True,
                                window = config['window'],
                                model = model,
                                tokenizer = tokenizer,
                                min_words = config['min_words'])

        # process output summary
        cur_i_summary = result[1]['S1_']
        if config['lowercase']:
            cur_summary += " " +  cur_i_summary.lower()
        
    
    out_str += '{}\n'.format(cur_summary)

    count += 1