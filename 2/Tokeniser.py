from transformers import AutoTokenizer, AutoModelForSequenceClassification


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# tokens = tokenizer.tokenize("hello world")
# print(tokens)

# tokens = tokenizer.convert_tokens_to_ids(tokens)
# print(tokens)

# tokens = tokenizer.encode("hello world")
# print(tokens)

model_inputs = ["I like cats.", "I like dogs and cats."]
data = tokenizer(model_inputs, padding=True, truncation=True, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 3)

output = model(**data)
print(output)


