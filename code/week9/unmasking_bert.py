import os
import torch
from transformers import BertTokenizer, BertForMaskedLM


# Guess what's behind the mask
def unmask_sent(text, top_k=5):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
  text = "[CLS] %s [SEP]"%text
  tokenized_text = tokenizer.tokenize(text)
  masked_index = tokenized_text.index("[MASK]")
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  tokens_tensor = torch.tensor([indexed_tokens])

  with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]  
  probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
  top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
    
  for i, pred_idx in enumerate(top_k_indices):
    predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
    if i==0:
      best_guess=predicted_token
    token_weight = top_k_weights[i]
    #print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
  return masked_index, best_guess   
  


user_input="go"

while user_input!="stop":
  print("#################### Unmasking with BERT #################\n")
  enter_input_text=input('Enter a sentence, use [MASK] for masking words: ').rstrip()
  if enter_input_text.split(" ")[-1]=="[MASK]":
    enter_input_text+=" [MASK]" # preventing punctuation predictions for final mask
  while "[MASK]" in enter_input_text:
    mask_id, unmasking=unmask_sent(enter_input_text)
    words=enter_input_text.split(" ")
    words[mask_id-1]=unmasking
    enter_input_text=' '.join(words)
  print("Unmasked:", enter_input_text)
  user_input=input("Proceed? (stop/go) ")





