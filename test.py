import torch
from CheXNetReportModel import CheXNetReportModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from create_datasets import test_dataset
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(41)

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2_prepared")
model = CheXNetReportModel(gpt2_model=gpt2)
model.load_state_dict(torch.load("model_epoch_9.pth", map_location=device))
model = model.to(device)
model.eval()

test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2_prepared")
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
print(f"Pad token id: {pad_token_id}, EOS token id: {eos_token_id}, BOS token id: {bos_token_id}")
max_len = 256

for i, batch in enumerate(test_dataloader):
    images = batch['images'].to(device)
    targets = batch['findings'].to(device)
    batch_size = images.size(0)

    input_ids = torch.full((images.size(0), 1), bos_token_id, dtype=torch.long).to(device)  # (batch_size, 1)
    predictions = []
    finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

    for _ in range(max_len):
        attn_mask = torch.ones(input_ids.size(), dtype=torch.long).to(device)

        outputs = model(images, input_ids, attn_mask)  # (batch_size, seq_len, vocab_size)

        logits = outputs[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = torch.where(finished, torch.full((batch_size,), pad_token_id, dtype=torch.long, device=device), next_token.squeeze(1))
        # Uncomment the following line to use argmax instead of sampling
        # next_token = torch.argmax(logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        finished |= next_token == eos_token_id

        if finished.all():
            break

    predictions = input_ids[:, 1:]  # remove bos

    # Convert predictions to text
    predictions = predictions.cpu().numpy()
    for j in range(batch_size):
        pred = predictions[j]
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        target_text = tokenizer.decode(targets[j], skip_special_tokens=True)
        print(f"Target report for batch {i} image {j}: {target_text}")
        print(f"Generated report for batch {i} image {j}: {pred_text}")
    # Stop after first batch for testing

    # if i == 0:
    #     break
