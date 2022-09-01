import pickle
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import math

# load news dataset
df = pickle.load(open('all_articles.pkl', 'rb'))
dataset = Dataset.from_pandas(df)
dataset = dataset.remove_columns(['stories_id', 'authors', 'publish_date', 'media_outlet'])
datasets = dataset.train_test_split(test_size=0.1)

# define tok function
def tokenize_function(examples):
    return tokenizer(examples["text"])

# define batching function
def chunk_examples(examples):
    chunks = []
    for sentence in examples['input_ids']:
        chunks += [sentence[i:i + 512] for i in range(0, len(sentence), 512)]
    return {'chunks': chunks}

def group_texts(examples, block_size=512):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# load pretrained BERT base
pretrained_model = "./pretrained/spanbert-large-cased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
print("begin tokenizing...")
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=datasets["train"].column_names)
print(tokenized_datasets)
# batch datasets
print("begin batching...")
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=4,
)

model = AutoModelForMaskedLM.from_pretrained(pretrained_model)

# training arguments
training_args = TrainingArguments(
    f"{pretrained_model}-finetuned-lbgtq-news",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# fine tune model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    data_collator=data_collator,
)
print("Begin finetuning")
trainer.train()

# evaluate
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

model.save_pretrained(f"{pretrained_model}-finetuned-lbgtq-news")
