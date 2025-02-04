from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import  Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
from datasets import load_dataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from accelerate import Accelerator

accelerator = Accelerator()

OUTPUT_DIR = './mbart-longer-train'

data_files = {
    'train': './data/smashed_train.csv',
    'validation': './data/smashed_val.csv'
}

dataset = load_dataset('csv', data_files=data_files)

print(dataset['train'][0])
print(dataset['validation'][0])

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,  # Use FP16 if training on a GPU
    logging_dir="./logs",  # Logging directory
    logging_steps=100,  # Log every 100 steps
    push_to_hub=False
)


def preprocess_function(examples):
    # Tokenize the input text
    inputs = tokenizer(examples["source_entity"], padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(examples["target"], padding="max_length", truncation=True, max_length=128)

    # Convert language codes to MBart token IDs
    lang_convert = {
        'ar_AE': 'ar_AR',
        'zh_TW': 'zh_CN',
        'es_ES': 'es_XX',
        'de_DE': 'de_DE',
        'fr_FR': 'fr_XX',
        'it_IT': 'it_IT',
        'ja_JP': 'ja_XX',
        'ko_KR': 'ko_KR',
        'tr_TR': 'tr_TR',
        'th_TH': 'th_TH'
    }
    lang_codes = [tokenizer.lang_code_to_id[lang_convert[lang]] for lang in examples["lang"]]

    # Attach labels and language information
    inputs["labels"] = targets["input_ids"]
    inputs["forced_bos_token_id"] = lang_codes  # Ensure model generates in the right language
    
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
