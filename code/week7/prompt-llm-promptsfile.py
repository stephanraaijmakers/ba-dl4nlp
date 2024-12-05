from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,  T5Tokenizer, T5ForConditionalGeneration, pipeline
import transformers
import torch

# Stephan Raaijmakers, December 2023

from random import shuffle

def read_prompts(fn):
    fp=open(fn,"r")
    prompts=[]
    prompt=""
    for line in fp:
        if line.rstrip()=="":
            if prompt!="":
                prompts.append(prompt)
                prompt=""
        else:
            prompt+=line.rstrip()+" "
    if prompt!="":
        prompts.append(prompt)
    return prompts

def select_prompts(prompts, n=10e4, permute=True):
    if permute:
        shuffle(prompts)
    return " ".join(prompts[:n])



def prompt_Falcon_zeroshot(fn_test):
    te_prompts=read_prompts(fn_test) # Only use "test" data, no training prompts    
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct") #, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    for prompt in te_prompts:
        sequences = pipe(
            prompt,
            max_length=200,
            # do_sample=True,
            # top_k=10,
            num_return_sequences=1,
            #eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")


def prompt_Falcon_nshot(fn_train, fn_test, n):
    tr_prompts=read_prompts(fn_train)
    train_prompt=select_prompts(tr_prompts, n, True)
    te_prompts=read_prompts(fn_test) # Only use "test" data, no training prompts

    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct") #, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    for prompt in te_prompts:
        print("Sending prompt:", prompt)
        sequences = pipe(
            train_prompt+"\n"+prompt,
            max_length=200,
            # do_sample=True,
            # top_k=10,
            num_return_sequences=1,
            #eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")








def Flan():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
    prompt = """Correct the following sentence: The today banks early close
    Output: Today the banks close early
    Correct the following sentence: A needs plant water
    Output:
    """

    prompt = """I be so happy when I wake up from a bad dream cus they be feelin too real.  
     Is a person who says this either brilliant, dirty, intelligent, lazy or stupid?
     """
    # answer produced: stupid

    prompt = """I am so happy when I wake up from a bad dream because they feel too real.  
    Is a person who says this either brilliant, dirty, intelligent, lazy or stupid?
            """
     # answer produced: intelligent
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids # CHANGE prompt
    outputs = model.generate(input_ids, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))



def Falcon():
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct") #, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    prompt = """Classify the text into neutral, negative or positive. 
    Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
    Sentiment:
    """

    prompt2 = """Identify the syntactic subject in the text.  
    Text: This movie is definitely one of my favorite movies of its kind. 
    Subject:
    """


    prompt3 = """Assign part of speech tags to the words in the text.  
    Text: This movie is great. 
    Parts of speech: This (article) movie (noun) is (verb) great (adverb)
    Text: I do not like this pizza, it is too salty.
    Parts of speech:
    """


    prompt4 = """Assign part of speech tags to the words in the text.  
    Text: This movie is great. 
    Parts of speech: This (article) movie (noun) is (verb) great (adverb)
    Text: The banks raise interest rates.
    Parts of speech:
    """

    prompt5 = """Evaluate the grammaticality of the text.  
    Text: This movie is great. 
    Grammaticality: grammatical
    Text: The today banks early close
    Grammaticality: ungrammatical
    Text: Today the banks close early
    Grammaticality: grammatical
    Text: The drove car fast too
    Grammaticality:
    Text: The car drove too fast
    Grammaticality:
    """

    prompt6 = """Evaluate the grammaticality of the text.  
    Text: The drove car fast too
    Grammaticality:
    """

    prompt7 = """Rewrite the following text.  
    Text: The language generated by LLMs has reached human parity: it is as good as or even better than what most humans would produce.
    """

    prompt8 = """Create a new text similar to the following text.  
    Text: Making money with bitcoins is a dubious activity.
    """

    prompt9 = """I am so happy when I wake up from a bad dream because they feel too real.  
    Is a person who says this either brilliant, dirty, intelligent, lazy or stupid?
            """
    
    prompt10 = """I be so happy when I wake up from a bad dream cus they be feelin too real.  
     Is a person who says this either brilliant, dirty, intelligent, lazy or stupid?
     """

    # Other
    sequences = pipe(
        prompt10,
        max_length=200,
        # do_sample=True,
        # top_k=10,
        num_return_sequences=1,
        #eos_token_id=tokenizer.eos_token_id,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    

if __name__=="__main__":
    #Falcon()
    #Flan()
    #prompt_Flan_zeroshot("./prompts_flan.txt")
    prompt_Falcon_nshot("./prompts_falcon_train.txt","./prompts_falcon_test.txt",5)
    exit(0)