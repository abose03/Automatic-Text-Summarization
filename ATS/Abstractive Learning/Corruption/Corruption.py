import random
import spacy
import pandas as pd
import pyinflect
from tqdm import trange

nlp = spacy.load("en_core_web_sm")

# try to identify the pronouns in the TEXT and replace with another pronoun in the "dic" dictionary
def replace_pro(text, dic):
    doc = nlp(text)

    def func(len, id):
        while True:
            r = random.randint(0, len-1)
            if r!=id:
                return r

    dic_1 = {}
    for key in dic.keys():
        if len(dic[key])>1:
            val = dic[key]
            for ind in range(len(val)):
                id = func(len(val), ind)
                dic_1[val[ind]] = val[id]

    #print(dic_1)

    words_tokenized = text.split()
    for i in range(len(words_tokenized)):
        if words_tokenized[i] in dic_1.keys():
            words_tokenized[i] = dic_1[words_tokenized[i]]

    text = " ".join(words_tokenized)
    return text


# delete the random words with given probability.
# Add noise -> add the same word twice or thrice. 
def delete_random_token(line, probability):
    """Delete random tokens in a given String with given probability
    Args:
        line: a String
        probability: probability to delete each token
    """
    
    def random_bool(probability=0.5):
        """Returns True with given probability
        Args:
            probability: probability to return True
        """
        assert (0 <= probability <= 1), "probability needs to be >= 0 and <= 1"
        return random.random() < probability
    
    line_split = line.split()
    ret = [token for token in line_split if not random_bool(probability)]
    
    def add_noise(text, percent=0.1):
        words = list(map(str, text.split()))

        # print random string
        i = int(percent * len(words))
        for _ in range(i):
            word = random.choice(words)

            index = words.index(word)
            #print(index)
            words.insert(index, words[index])

        return " ".join(words)
    
    return add_noise(" ".join(ret), percent=0.1)


# Identify the entities like person, place, numbers, Currencies etc and replace them with corresponding entities.
def show_ents(doc1): 
    df=pd.DataFrame()
    l=[]
    l1=[]
    #org.append(doc1)
    doc = nlp(doc1)
    if doc.ents: 
        for ent in doc.ents: 
            # print("Entity : " +ent.text)
            # Print("Entity Label :"+ent.label_+ ' - '+str(spacy.explain(ent.label_)))
            l.append(ent.label_)
            l1.append(ent.text)
    #else: print('No named entities found.')
    df['ent_lab']=l
    df['ent']=l1
    df1 = df['ent_lab'].value_counts().rename_axis('unique_values').reset_index(name='counts')

    if (df['ent_lab'].value_counts()>1).any():
        #print('if')
        text=doc1[:]
        x=df1.unique_values[df1.counts>1]
        for i in range(0,len(x)):
            y=df.ent.loc[df.ent_lab==x[i]]
            y=y.reset_index(drop=True)
            #print(y)
            # print(type(text))
            text=text.replace(y[0],'mm').replace(y[1],y[0]).replace('mm',y[1]) 
        #print(text)
        #swapped.append(text)

    else:
        # y=df1.unique_values[df1.counts==1]
        #print('else')
        try: 
            text=doc1[:]
            doc_dep = nlp(text)
            for i in range(len(doc_dep)):
                token = doc_dep[i]
                # print(token.tag_)
                if token.tag_ in ['VBP','VBZ','VBD']:
                    # print(token.text, token.lemma_, token.pos_, token.tag_) 
                    text = text.replace(token.text, 'did not'+ ' ' +token.text)
                    text = text.replace(token.text, token._.inflect("VB"))
                    # text = text.replace(token.text, 'not'+token.text)
            #print(text)
            #swapped.append(text)
        except:
            return doc1
    return text

dic = {'Art': ['the', 'a', 'an'],
 'Prs': ['him',
  'he',
  'it',
  'her',
  'his',
  'they',
  'its',
  'their',
  'our',
  'you',
  'them',
  'she',
  'my',
  'your',
  'we',
  'himself',
  'i',
  'me',
  'herself',
  'us',
  'itself',
  'yourself',
  'themselves',
  'hers',
  'mine',
  'yours',
  'theirs',
  'ourselves',
  'myself'],
    'Dem': ['that', 'this', 'then', 'those', 'here', 'there', 'these']}

###################################################
# load the data frome excel or csv
train_df = pd.read_excel("../DATA/BART_Test_Merged_corrupted.xlsx") 
# drop null rows if any
train_df.dropna(inplace=True)
print(train_df.info())


swapped, replaced, deleted= [], [], [], [], [], []

for i in trange(len(train_df)):
        
    # take the text for which we need corruption.
    text=str(train_df.iloc[i]['distilbart-cnn-12-6']).lower()
    
    # calling the show_ents function which will swap entities like name, place etc.
    entity_num_swap = show_ents(text)
    
    # calling replace_pro function which will replace the pronouns in the text with random pronouns in the "dic"
    replaced_text = replace_pro(text, dic)
    
    # Calling the delete_random_token function which will delete random words and also add few noise.
    deleted_text = delete_random_token(text, probability=0.2)
    
    deleted.append(deleted_text)
    replaced.append(replaced_text)
    swapped.append(entity_num_swap)
    

# creaking separate column for all three corruptions
train_df['CBART_noisy'] = swapped
train_df['CBART_Pronoun_replaces'] = replaced
train_df['CBART_deleted_added_noise'] = deleted

# saving the file to excel or csv.
train_df.to_excel("../DATA/BART_Test_Merged_corrupted.xlsx", index=False)

