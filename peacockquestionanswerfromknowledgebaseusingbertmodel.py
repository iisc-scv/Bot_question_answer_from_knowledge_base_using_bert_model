# -*- coding: utf-8 -*-
# @uthor Subhash Chandra

!pip install transformers

import torch

from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = "How many parameters does BERT-large have?"
answer_text = "BERT-large is really big... it has 24 layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."

# Apply the tokenizer to the input text, trating then as a text-pair.
input_ids = tokenizer.encode(question, answer_text)
print('The input has a total of {:} tokens.'.format(len(input_ids)))

#BERT only needs the token IDs, but for the purpose of inspecting the 
# tokenizer's behavior, let's also get the token strings and display them. 
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# For each token and its id..
for token, id in zip(tokens, input_ids):
    # If this is the [SEP] token, add some space around it to make it stand out. 
    if id == tokenizer.sep_token_id:
      print('')
    #Print the token string and its ID in two columsn. 
    print('{:<12} {:>6}'.format(token, id))

    if id == tokenizer.sep_token_id:
      print('')

# Search the input_ids for the first instance of the '[SEP]' token.
sep_index = input_ids.index(tokenizer.sep_token_id)

# The number of segment A tokens includes the [SEP] token itself.
num_seg_a = sep_index + 1

#The remainder are segment B.
num_seg_b = len(input_ids) - num_seg_a

#Construct the list of 0s and 1s.
segment_ids = [0]*num_seg_a + [1]*num_seg_b

# There should be a segment_id for every input token.
assert len(segment_ids) == len(input_ids)

# Run model for testing 
start_scores, end_scores = model(torch.tensor([input_ids]), )   #The tokens representing our iput text.
token_type_ids = torch.argmax(end_scores)

#Find the token wiht the highest 'start' and 'end' scores.
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

# Combine the tokens in the answer and print it out.ndin
answer = ''.join(tokens[answer_start:answer_start+1])

print('Answer: "' + answer + '"')

# Start wiht the first token. 
answer = tokens[answer_start]

# Select the remaining answer tokens and join them with whitespace.
for i in range(answer_start + 1, answer_end + 1):
             
             #If it's a subword token, then recombine it with the previous token.
             if tokens[i][0:2] == '##':
               answer += tokens[i][2:]
             
             # Otherwise, add a space then the token.
             else:
                  answer += ' ' + tokens[i]
             print('Answer: "' + answer + '"')

import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size
# sns.set(font_scale*1.5)
plt.rcParams["figure.figsize"] = (16,8)

# Pull the scores out of PyTorch tensors and convert them to 1D numpy arrays.
s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()

# We'll use the tokens as the x-axis labels. In order to do that, they all need
# to be unique, so we'll add the token index to the end of each one.
token_labels = [] 
for (i, token) in enumerate(token):
    token_labels.append('{:} - {:>2}'.format(token, i))

# Create a barplot showing the start word score for all of the tokens.
#ax = sns.barplot(x=token_labels, y=e_scores, ci=None)

# Turn the xlabels vertical.
#ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, ha="center")

#Turn on the vertical grid to help align words to scores. 
#ax.grid(True)

#plt.title('Start word Scores')

#plt.show()

import pandas as pd

# Stre the tokens and scores in a DataFrame. 
# Each token will have two rows, one for its start score and one for its ends.
# score. The "marker" column will dirrerentiate them. A little wacky, I know.
scores = []  
for (i, token_label) in enumerate(token_labels):

       # Add the token's start score as one row. 
       scores.append({'token_label':token_label,
                      'score': s_scores[i],
                      'marker': 'start'})
       # Add the token's end spcore as another row.
       scores.append({'token_label': token_label, 
                     'score':e_scores[i],
                     'marker':'end'})   
       
       df = pd.DataFrame(scores)

#Draw a grouped barplot to show start and end scores for each word. 
# The "hue" parameter is where we tell it which datpoints belong to which
#of the ktwo series. 
g = sns.catplot(x="token_label", y = "score", hue = "marker", data = df,
                 kind = "bar", height = 6, aspect =4)

# Turn the xlabels vertical 
g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align word to scores.
g.ax.grid(True)

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ==============Tokenize===========
    # Apply the tokenize to the iput text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.


    #changes are made to test number of tokens 
    #print('Query has {:,} tokens.\n'.format(len(input_ids)))

    #========Set Segment IDs ==================
    #Search the input_ids for the first instance of the '[SEP] token itself.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token itself
    num_seg_a = sep_index + 1

    # The remainder are segment B. 
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    #There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ================ Evaluate ===============
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]),  # The tokens repersenting the input.
                                      token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
                          
    # =================== Reconstruct Answer =================
    # Find the tokens with the highest 'start' and 'end' scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string version of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
          
          # If it's a subword token, then recombine it with the previous.
          if tokens[i][0:2] == '##':
             answer += tokens[i][2:]

          # Otherwise, add a space then the token.
          else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')

import textwrap

# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80)

bert_abstract = '''


Friction is the resistance to motion of one object moving relative to another. When surfaces in contact move relative to each other, the friction between the two surfaces converts kinetic energy into thermal energy. Friction is a non fundamental and non conservative force.


A coefficient of friction is a dimensionless scalar value that shows the relationship between two objects and the normal reaction between the objects that are involved.

There are two main types of friction, static friction and kinetic friction. 
Static friction is friction between two or more solid objects that are not moving relative to each other. Kinetic friction, also known as dynamic friction or sliding friction, occurs when two objects are moving relative to each other and rub together.


The limiting maximum static frictional force depends upon the nature of the surfaces in contact. It does not depend upon the size or area of the surfaces. For the given surfaces, the limiting frictional force f s is directly proportional to the normal reaction N. F s is equal to mu s multiplied by N. Where the constant of proportionality mu s is called the coefficient of static friction. The said formula holds only when f s has its maximum, limiting value.

In the case of limiting friction, the angle which the resultant R of the limiting force f s and the normal reaction N makes with the normal reaction N is called the angle of friction.


A Force is that physical cause which changes or tends to change either the size or shape or the state of rest or motion of the body.


The forces which act on bodies when they are in physical contact are called the Contact forces.

The forces experienced by bodies even without being physically touched are called the Non-Contact Forces or the Forces at a distance.

The body on a force due to the earth's attraction is called the force of Gravity. In the universe, all particles attract one another due to its mass.

The turning effect of a force acting on a body about an axis is due to the moment of force or torque. It depends on the magnitude of a force applied and the distance of the line of action of force from the axis of rotation.


'''

question = "What is friction "

answer_question(question, bert_abstract)

