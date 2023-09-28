import numpy as np
from torchinfo import summary
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import load_dataset, load_metric

def tokenize_fn(batch):
    return tokenizer(batch["sentence"], truncation=True)

def compute_metrics(logits_and_labels):
    metric = load_metric("glue", "sst2")
    logits, labels = logits_and_labels
    predicitions = np.argmax(logits, axis=-1)
    return metric.compute(predicitions, labels)


checkpoint = "distilbert-base-uncased"
raw_datasets = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_dataset = raw_datasets.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
    'my_trainer',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=5,
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(model, training_args, train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["validation"], tokenizer=tokenizer, compute_metrics=compute_metrics)

trainer.save_model('./models/glue_bert')


