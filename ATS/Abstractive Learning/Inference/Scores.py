from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import pandas as pd
from tqdm import trange
from statistics import mean
import os

# load the csv or excel file which has Actual Summaries and Predicted Summaries.
PATH = "../DATA/BART_Predictions_test_article_BART.xlsx"

df = pd.read_excel(PATH)
print(df.columns)

# Assign Acutal Summaries to "Actual_summaries" which will be a list.
Actual_summaries = df['highlights']
# Assign Predicted Summaries to "Predicted_summaries" which will be a list.
Predicted_summaries = df['Predictions']


# Checking if both Actual_summaries and Predicted_summaries has same length.
assert len(Actual_summaries) == len(Predicted_summaries)


Print_string = ""
#######################    BLEU SCORE CALCULATION   #####################

#Ref, cand = ["The cat is on the mat", "There is a cat on the mat"] , "The cat the cat is on the mat"

print("Calculationg BLEU")

Bleu_Score = []
for i in trange(len(Actual_summaries)):
    
    try:
        Ref = [Actual_summaries[i]]
        cand = Predicted_summaries[i]

        ref = [' '.join(ref.split(' ')).lower().split() for ref in Ref]
        cand = ' '.join(cand.split(' ')).lower().split()
        
        # calculate the BLEU score for ref(ground-truth-labels) and cand(predictions) 
        # score will be calculated by taking    0.5 weightage for 1-gram 
        #                                       0.25 weightage for 2-gram 
        #                                       0.25 weightage for 3-gram
        score = sentence_bleu(ref, cand, weights=(0.5, 0.25, 0.25, 0))
        
    except:
        score = 0
        
    Bleu_Score.append(score)


# checking if Bleu_scores list has same length as Actual_summaries
assert len(Bleu_Score) == len(Actual_summaries) 

# Creating a separate columns for Bleu Score.
df["Bleu_Score"] = Bleu_Score

# Creating the String to print on screen which contains all Scores BLEU, ROUGE, BERT
Print_string += " "*5 + "BLEU_SCORE RESULT" + " "*5 + "\n"
Print_string += f"AVG_SCORE - {mean(Bleu_Score)}" + "\n\n"

##############################################################################


#######################    ROUGE SCORE CALCULATION   ########################
print("Calculationg ROUGE")

# creating rouge_scorer.RougeScorer object providing which n-grams we need the score for.
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
# scorer.score(target, predicted)

Rouge_Score_1 = []
Rouge_Score_2 = []
Rouge_Score_3 = []
Rouge_Score_L = []
for i in trange(len(Actual_summaries)):
    
    try:
        Ref = Actual_summaries[i]
        cand = Predicted_summaries[i]

        # calcualte the score for the ref(ground-truth-label) and cand(Prediction)
        scores = scorer.score(Ref, cand)
        
        #print(scores.fmeasure)
        Rouge_Score_1.append(scores['rouge1'].fmeasure)
        Rouge_Score_2.append(scores['rouge2'].fmeasure)
        Rouge_Score_3.append(scores['rouge3'].fmeasure)
        Rouge_Score_L.append(scores['rougeL'].fmeasure)
        #Rouge_Score.append(scores)
        
    except:
        Rouge_Score_1.append(0)
        Rouge_Score_2.append(0)
        Rouge_Score_3.append(0)
        Rouge_Score_L.append(0)
    
    
# checking if Rouge Scores list has same length as Actual_summaries
assert len(Rouge_Score_1) == len(Actual_summaries)  
assert len(Rouge_Score_2) == len(Actual_summaries) 
assert len(Rouge_Score_3) == len(Actual_summaries) 
assert len(Rouge_Score_L) == len(Actual_summaries) 

# Creating a separate column for all rouge scores.
df["Rouge_Score_1"] = Rouge_Score_1
df["Rouge_Score_2"] = Rouge_Score_2
df["Rouge_Score_3"] = Rouge_Score_3
df["Rouge_Score_L"] = Rouge_Score_L

Print_string += " "*5 + "ROUGE_SCORE RESULTS" + " "*5 + "\n"
Print_string += f"AVG_ROUGE 1 SCORE - {mean(Rouge_Score_1)}" + "\n"
Print_string += f"AVG_ROUGE 2 SCORE - {mean(Rouge_Score_2)}" + "\n"
Print_string += f"AVG_ROUGE 3 SCORE - {mean(Rouge_Score_3)}" + "\n"
Print_string += f"AVG_ROUGE L SCORE - {mean(Rouge_Score_L)}" + "\n\n"

#############################################################################
os.system("clear")
print(Print_string)

#######################   BLEU SCORE CALCULATION   ##########################

print("Calculationg BERT SCORE")
scorer = BERTScorer(lang="en", rescale_with_baseline=True)

cands = Predicted_summaries
# a sentence with nothing is giving error so for that cases we are taking "^" instead of null
cands = ["^" if str(i).strip()=='' else str(i) for i in Predicted_summaries]

refs = Actual_summaries
ref = ["^" if str(i).strip()=='' else str(i) for i in Actual_summaries]
refs = [[i] for i in refs]

# Calculate scores using the cands and refs.
P, R, F1 = scorer.score(cands, refs)
#scorer.plot_example(cands[0], refs[0])

assert len(F1) == len(Actual_summaries) 
df['BERT_F1'] = F1

Print_string += " "*5 + "BERT_SCORE RESULTS" + " "*5 + "\n"
Print_string += f"AVG_BERT PRECISION - {mean(P.numpy())}" + "\n"
Print_string += f"AVG_BERT RECALL - {mean(R.numpy())}" + "\n"
Print_string += f"AVG_BERT F1 - {mean(F1.numpy())}" + "\n"
############################################################################

# Clearing the screen completely and printing the Scores BLEU, ROUGE and BERT.
os.system("clear")
print(" "*10, PATH, "\n")
print(Print_string)

# saving Scores.
df.to_excel("BART_Scores.xlsx", index=False)