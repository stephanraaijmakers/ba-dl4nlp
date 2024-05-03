# !pip install llamaapi
# Stephan Raaijmakers, 2024

from itertools import product, combinations, chain
from more_itertools import pairwise
import itertools
import re
import json
from llamaapi import LlamaAPI


import spacy
nlp = spacy.load("en_core_web_sm")

# Register with LLAMAAPI at https://www.llama-api.com
# It is free with a credit of $5 and price/token is very low.
# Experiment with a few of their models (keep an eye on your Usage).

llama = LlamaAPI(LLAMAAPI_KEY) # bad practice, delete key from your code after testing

# After trying out the Game24 example: can you implement this function? It extracts entities with labels from a string.
def get_entities(text): # string, like "Net income was $9.4 million compared to the prior year of $2.7 million."
  nerD={}
  for doc in nlp.pipe([text], disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
    for x in [(ent.text, ent.label_) for ent in doc.ents]:
      nerD[x[0]]=x[1] 
  # Generate json
  return json.dumps(nerD) 

                
def game24(x1,x2,x3,x4):
    operators=['+','-','/','*']
    OP=[]
    for o1 in operators:
        for o2 in operators:
            for o3 in operators:
                OP.append([o1,o2,o3])
    perm=list(itertools.permutations([x1,x2,x3,x4]))
           
    for p in perm:
        for op in OP:
            e=[str(p[0])+op[0],str(p[1])+op[1],str(p[2])+op[2],str(p[3])]

            def all_bracketings(seq):
                if len(seq) <= 1:
                    yield from seq
                else:
                 for n_children in range(2, len(seq)+1):
                     for breakpoints in combinations(range(1, len(seq)), n_children-1):
                        children = [seq[i:j] for i,j in pairwise(chain((0,), breakpoints, (len(seq)+1,)))]
                        yield from product(*(all_bracketings(child) for child in children))

            br=list(all_bracketings(e))
            for b in br:
                b=str(b)
                orig=b
                while True:                    
                    b=re.sub(",","",b)
                    b=re.sub("'","",b)
                    b=re.sub("\+\)",")+",b)
                    b=re.sub("\-\)",")-",b)
                    b=re.sub("/\)",")/",b)
                    b=re.sub("\*\)",")*",b)
                    if b!=orig:
                        orig=b
                    else:
                        break
                try:
                    if eval(b)==24:
                        solution={"solution":b}
                        return json.dumps(solution)
                except ZeroDivisionError:
                    True
    return(json.dumps({"solution":"None"}))
                  

    
# Results differ across and within models, so repeat a few times with same model/other models.
    
        
# First call
# =============================
                    
api_request_json = {
    #"model": "llama-13b-chat",
    "model": "mixtral-8x22b-instruct",
    "messages": [
        {"role": "user", "content": "Solve the Game24 puzzle for these numbers: 10 12 3 8"},
    ],
    "functions": [
        {
            "name": "game24",
            "description": "Solve the Game24 puzzle",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {
                        "type": "number",
                        "description": "The first number",
                    },
                    "x2": {
                        "type": "number",
                        "description": "The second number",
                    },
                    "x3": {
                        "type": "number",
                        "description": "The third number",
                    },
                    "x4": {
                        "type": "number",
                        "description": "The fourth number",
                    },
                },
            },
            "required": ["x1","x2","x3","x4"],
        }
    ]
}

response = llama.run(api_request_json)
output = response.json()['choices'][0]['message']
output["content"]=""
print(output)

# {'role': 'assistant', 'content': '', 'function_call': {'name': 'game24', 'arguments': {'x1': 10, 'x2': 12, 'x3': 3, 'x4': 8}}}


response = llama.run(api_request_json)
output = response.json()['choices'][0]['message']

solution=game24(10,12,3,8) # extract from the output of the first API call
print(solution)

second_api_request_json = {
    #"model": "llama-13b-chat",
    "model": "mixtral-8x22b-instruct",
    "messages": [
      {"role": "user", "content":"Solve the Game24 puzzle for these numbers: 10 12 3 8"},
      {"role": "function", "name": output['function_call']['name'], "content": solution}
    ],
    "functions": [
        {
            "name": "game24",
            "description": "Solve the Game24 puzzle",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {
                        "type": "number",
                        "description": "The first number",
                    },
                    "x2": {
                        "type": "number",
                        "description": "The second number",
                    },
                    "x3": {
                        "type": "number",
                        "description": "the third number",
                    },
                    "x4": {
                        "type": "number",
                        "description": "The fourth number",
                    },
                },
            },
            "required": ["x1","x2","x3","x4"],
        }],
  }

second_request = llama.run(second_api_request_json)
summarizedResponse = llama.run(second_api_request_json)
print(summarizedResponse.json())
print("The solution is: ",summarizedResponse.json()['choices'][0]['message']['content'])



# {'created': 1714653169, 'model': 'llama-13b-chat', 'usage': {'prompt_tokens': 190, 'completion_tokens': 58, 'total_tokens': 248}, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The solution to the Game24 puzzle for the given numbers 10, 12, 3, and 8 is:\n\n(10 * (12 / (8 - 3))) = 10 * (12 / 5) = 10 * 2.4 = 24', 'function_call': None}, 'finish_reason': 'max_token'}]}


    
        
        
    
