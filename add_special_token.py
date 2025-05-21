from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'additional_special_tokens': ['[REDACTED]']}) # we replaced XXXX with [REDACTED]

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

model.save_pretrained('gpt2_prepared')
tokenizer.save_pretrained('gpt2_prepared')