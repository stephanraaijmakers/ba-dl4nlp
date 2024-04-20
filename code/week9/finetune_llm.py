# !pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
import os
import torch
import gc
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

def clean_memory():
  del model
  del pipe
  del trainer
  gc.collect()
  gc.collect()


def finetune():
    #model_name = "NousResearch/Llama-2-7b-chat-hf"
    #model_name="Maykeye/TinyLLama-v0"
    #model_name="keeeeenw/MicroLlama"
    model_name="vikash06/llama-2-7b-small-model-new"
    #model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.6"
    dataset_name = "prompts.txt"
    #"mlabonne/guanaco-llama2-1k"

    # Fine-tuned model
    new_model = "llama-2-7b-dl4nlp"

    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    output_dir = "./results"
    num_train_epochs = 1

    # Set bf16 to True with an A100
    fp16 = False
    bf16 = False

    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
    max_grad_norm = 0.3 # gradient clipping
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1
    warmup_ratio = 0.03
    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True
    # Save checkpoint
    save_steps = 0
    # Log update steps
    logging_steps = 25

    # SFT parameters
    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False
    # Load the entire model on the GPU 0
    device_map = {"": 0}

    # Training
    dataset = load_dataset("text", data_files={"train": ["drive/MyDrive/finetuning_llm/data/prompts.txt"], "test": "drive/MyDrive/finetuning_llm/data/prompts.txt"})
    dataset_train=dataset["train"]

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("GPU supports bfloat16: accelerate training with bf16=True")


    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_train,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model)

    logging.set_verbosity(logging.CRITICAL)

    # Text generation
    print("#"*80)
    print("Old model:")
    prompt = "How to own a plane?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

    del model
    del pipe
    del trainer
    gc.collect()
    gc.collect()

    #clean_memory()

    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
      model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()

    print("#"*80)
    print("New model:")
    prompt = "How to own a plane?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])


    # Reload tokenizer to save it
    #tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.padding_side = "right"
    #   !huggingface-cli login
    #   model.push_to_hub(new_model, use_temp_dir=False)
    #   tokenizer.push_to_hub(new_model, use_temp_dir=False)


if __name__=="__main__":
    finetune()
