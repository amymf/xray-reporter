from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add padding token and [REDACTED] token
tokenizer.add_special_tokens({
    'pad_token': '[PAD]',
    'additional_special_tokens': ['[REDACTED]']
})

# Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Resize token embeddings to include new special tokens
model.resize_token_embeddings(len(tokenizer))

# Save the updated tokenizer and model
model.save_pretrained('gpt2_prepared')
tokenizer.save_pretrained('gpt2_prepared')
