#  importing all the libraries required 
import os
import json
import time
import torch
import random 
import numpy as np
import pandas as pd
from statistics import mean
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoConfig, AutoTokenizer, GPTNeoForCausalLM, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm import tqdm

# importing nlp for calculating bleu score in validation loop
import nlp
bleu_metric = nlp.load_metric('bleu')

# To avoid printing warning messages while training.
import warnings
warnings.filterwarnings("ignore")

# making tokenizer parallelism to false
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# importing wandb and login. For generating loss graphs.
import wandb
wandb.login()

# initialize a project in your account in wandb.
num = 1
wandb.init(project="GPTNeo-CNN-Summarization_BART_EXtracts", name=f"Exp-{num}")

# loading train and validation data.
df_train = pd.read_excel('../DATA/BART_Train_Merged_corrupted.xlsx')
df_val = pd.read_excel('../DATA/BART_Validation_Merged_corrupted.xlsx')
print(df_train.shape, df_val.shape)

# seeding everything.
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
    
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 2021
seed_everything(seed)

###############################

# entering all hyperparameters in this block.

epochs=30
train_batch_size=2
val_batch_size=2
src_max_length=1000
tgt_max_length=500
warmup_prop = 0.01
lr = 1e-6

################################

wandb.config = {
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": train_batch_size
               }


model_name = 'EleutherAI/gpt-neo-125M'
# loading tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name)


# class for our custom dataset.
class GPTNeoSummDataset(Dataset): 
    
    def __init__(self, text, summarized_text, corrupted_text, pronoun_swap, deleted_added_noise, 
             tokenizer, src_max_length, tgt_max_length):
        
        
        
        self.text = text
        self.summarized_text = summarized_text
        self.corrupted_text = corrupted_text
        self.pronoun_swap = pronoun_swap
        self.deleted_added_noise = deleted_added_noise

        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        
        # creating a dictionary named "dic" for corrupted text 
        #        0 -> entity swap
        #        1 -> pronoun swap
        #        2 -> deleted and added noise.
        self.dic = {
                    0:self.corrupted_text, 
                    1:self.pronoun_swap, 
                    2:self.deleted_added_noise
                }


    def __len__(self):
        return len(self.pronoun_swap)

    def __getitem__(self, item):
        
        # input to the model.
        text = self.text[item]
        
        # ground truth label.
        summarized_text = self.summarized_text[item]
        
        # input format to pass through to model.
        input_text = 'summarize: ' + str(text) + "\n" +  str(summarized_text)

        # tokenizing the model input text.
        inputs = self.tokenizer(input_text, 
                                 None, 
                                 add_special_tokens=True,
                                 max_length=self.src_max_length,
                                 truncation = True,
                                 padding = 'max_length',
                                 return_tensors = 'pt'
                                )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

         # tokenizing the ground truth labels.
        outputs = self.tokenizer(input_text, 
                                 None, 
                                 add_special_tokens=True,
                                 max_length=self.src_max_length,
                                 truncation = True,
                                 padding = 'max_length',
                                 return_tensors = 'pt'
                                )

        output_ids = outputs["input_ids"].squeeze(0)
        output_ids[output_ids==self.tokenizer.pad_token_id]=-100

        # selecting a number from 0-2 randomly and taking that value in self.dic 
        #        0 -> entity swap
        #        1 -> pronoun swap
        #        2 -> deleted and added noise.
        corrupted_text = self.dic[random.randint(0, 2)][item]
        corrupted_text = 'summarize: ' + str(text) + "\n" + str(corrupted_text)
        
        # tokenizing the corrupted text.
        outputs_corrupted = self.tokenizer(corrupted_text, 
                                 None, 
                                 add_special_tokens=True,
                                 max_length=self.src_max_length,
                                 truncation = True,
                                 padding = 'max_length',
                                 return_tensors = 'pt'
                                )

        output_ids_corrupted = outputs_corrupted["input_ids"].squeeze(0)
        
        # indexed to ignore while calculating the cross-entropy loss.
        output_ids_corrupted[output_ids_corrupted==self.tokenizer.pad_token_id]=-100
        
        return {
            'text':input_text,
            'summarized_text':summarized_text,
            'corrupted_text':corrupted_text,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'output_ids': torch.tensor(output_ids, dtype=torch.long),
            'output_ids_corrupted': torch.tensor(output_ids_corrupted, dtype=torch.long)
        }


# creating the dataset for training 
#       text  ->  input to model.
#       Summarized_text  ->  ground truth labels for training.
#       corrupted_text  ->  entity swap
#       pronoun_swap    ->  pronoun swap
#       deleted_added_noise  ->  added and deleted noise.
train_dataset = GPTNeoSummDataset(text=df_train['distilbart-cnn-12-6'].values,
                              summarized_text=df_train['highlights'].values,
                              corrupted_text=df_train['noisy'].values,
                              pronoun_swap = df_train['Pronoun_replaced'].values, 
                              deleted_added_noise = df_train['deleted_added_noise'].values,
                              tokenizer=tokenizer,
                              src_max_length=src_max_length,
                              tgt_max_length=tgt_max_length
                             )

# creating the dataset for validation.
val_dataset = GPTNeoSummDataset(text=df_val['distilbart-cnn-12-6'].values,
                            summarized_text=df_val['highlights'].values,
                            corrupted_text=df_val['noisy'].values,
                            pronoun_swap = df_val['Pronoun_replaced'].values, 
                            deleted_added_noise = df_val['deleted_added_noise'].values,
                            tokenizer=tokenizer,
                            src_max_length= src_max_length,
                            tgt_max_length= tgt_max_length
                             )

# creating the Dataloader for training. using batch_size.
train_dataloader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              drop_last=False,
                              shuffle=True,
                              num_workers=2)

# creating the Dataloader for validation. using batch_size.
val_dataloader = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            drop_last=False,
                            num_workers=2)


# assigning device to GPU if available else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# loading the config for model.
config = GPTNeoConfig.from_pretrained(model_name)
#config

# loading the Pretrained model.
model = GPTNeoForCausalLM.from_pretrained(model_name,
                                          return_dict=True
                                          )
model.config.pad_token_id = model.config.eos_token_id

# moving to device.
_ = model.to(device)

# calculating steps.
length_loader = len(df_train)/train_batch_size
num_warmup_steps = int(warmup_prop * epochs * length_loader )
num_training_steps = epochs * length_loader
print("num_training_steps: ", num_training_steps) 

# creating optimizer and scheduler.
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)


alpha = 0.1
M = 80

# loss function -> crossentropy.
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)


def training_loop():
    
    model.train()
    train_loss=[]
    bi = 0
    
    pbar = tqdm(train_dataloader, desc="Training:. ")
    
    # load each batch one by one.
    for d in pbar:
        
        input_ids = d["input_ids"]
        attention_mask = d["attention_mask"]
        output_ids = d["output_ids"]
        output_ids_corrupted=d["output_ids_corrupted"]

        # moving to device 
        input_ids = input_ids.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        output_ids = output_ids.to(device, dtype=torch.long)
        output_ids_corrupted=output_ids_corrupted.to(device,dtype=torch.long)
        
        # zeroing the gradients.
        optimizer.zero_grad()
        
        # forward pass the model
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=output_ids,   
                        return_dict=True,
                        output_hidden_states=True
                       )
        
        
        # getting logits.
        logits=outputs.logits
        
        #lm_logits = lm_logits.to(torch.float32)
        logits = logits.to(torch.float32)

          # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_output_ids = output_ids[..., 1:].contiguous()
        shift_output_ids_corrupted = output_ids_corrupted[..., 1:].contiguous()
        
          # Flatten the tokens
        #loss_fct = CrossEntropyLoss()
        #loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))     
        
        # calculating loss1 with respect to pred and original ground truth label.
        loss1 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_output_ids.view(-1))
        
         # calculating the loss2 with respect to pred and corruption data.
        loss2 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_output_ids_corrupted.view(-1))
        
        #lm_logits = lm_logits.to(hidden_states.dtype)
        logits = logits.to(outputs['hidden_states'][0].dtype)
        
        # final loss
        #loss = loss1 - alpha*loss2   # 1st type of loss
        loss = loss1 + alpha * max(loss1 + M - loss2, 0)   # 2nd type of loss
        
        
        #loss = loss.to(hidden_states.dtype)
        loss = loss.to(outputs['hidden_states'][0].dtype)
        
        #pbar.set_postfix({'Loss_1': round(loss1.item(), 4)})
        
        # loggin the losses to wandb.
        wandb.log({'train_loss_1': loss1.item()})
        wandb.log({'train_loss_2': loss2.item()})
        wandb.log({'loss': loss.item()})
        
        # logging the losses to the progress bar.
        pbar.set_postfix({'Loss': round(loss.item(), 4),
                          'Loss_1': round(loss1.item(), 4), 
                          'Loss_2': round(loss2.item(), 4)})


        train_loss.append(loss.item())
        
        # backward pass and weights updation.
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        bi+=1

    # returning final average training loss.
    return mean(train_loss) 


def validation_loop():
    
    model.eval() 
    
    bleu = []
    
    # specifying not to calculate gradients.
    with torch.no_grad():
        bi = 0
        
        # going through all batches.
        for d in tqdm(val_dataloader):
            
            text = d["text"]
            summarized_text = d['summarized_text']
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            output_ids = d["output_ids"]
            output_ids_corrupted=d["output_ids_corrupted"]

            # moving to device.
            input_ids = input_ids.to(device, dtype=torch.long)
            attention_mask = attention_mask.to(device, dtype=torch.long)
            output_ids = output_ids.to(device, dtype=torch.long)
            output_ids_corrupted=output_ids_corrupted.to(device, dtype=torch.long)

            # generating the summary for the input article.s
            preds_ids = model.generate(input_ids, max_length=2000, num_return_sequences=1)
            
            # for all articles in this batch calculate bleu score. if pred in not empty string else bleu is zero.
            for i in range(preds_ids.shape[0]):
                
                preds = tokenizer.decode(preds_ids[i].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
                #print(preds)
                if preds.strip() != "":
                    score = bleu_metric.compute([preds.split()], [[summarized_text[0].split()]])['bleu']
                    bleu.append(score)
                else:
                    bleu.append(0)

            bi+=1
            
    #return mean(bleu) 
    return mean(bleu)


# assigning first bleu score as -infinity.
final_bleu_score = float("-inf")

# doining for epochs num of times.
for i in range(epochs):
    
    # training.
    loss = training_loop()
    
    # validation.
    bleu_score = validation_loop()
    print(f"Epoch: {i}, Bleu_score: {bleu_score}")
    
    # logging the validation loss to wandb.
    wandb.log({'Bleu_score': bleu_score})
        
    # saving the model if current bleu score is best than final bleu score.
    if bleu_score > final_bleu_score:
        print("Saving the model")
        final_bleu_score = bleu_score 
        model.save_pretrained("GPT_Model")
    