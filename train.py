from transformers import GPT2LMHeadModel, GPT2Tokenizer
from CheXNetReportModel import CheXNetReportModel
import torch 
# import wandb
from torch.utils.data import DataLoader
from create_datasets import train_dataset, val_dataset

# wandb.init(project="CheXNet-report-generation")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(41)

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2_prepared")

model = CheXNetReportModel(gpt2_model=gpt2)
model = model.to(device)
model.train()

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2_prepared")
pad_token_id = tokenizer.pad_token_id
redacted_id = tokenizer.convert_tokens_to_ids('[REDACTED]')

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        images = batch['images'].to(device) # (batch_size, N, num_channels, H, W) where N is the number of images
        batch_size, N, num_channels, H, W = images.shape
        images = images.view(batch_size * N, num_channels, H, W) # (batch_size * N, num_channels, H, W)
        findings = batch['findings'].to(device) # (batch_size, seq_len)
        attn_mask = batch['attn_mask'].to(device) # (batch_size, seq_len)

        findings = findings.masked_fill(findings == redacted_id, pad_token_id)  # replace [REDACTED] with pad token
        outputs = model(images, findings, attn_mask) # (batch_size, seq_len, vocab_size)
        
        vocab_size = outputs.size(-1)
        o = outputs.view(-1, vocab_size) # (batch_size * seq_len, vocab_size)
        target = findings.view(-1) # (batch_size * seq_len)
        loss = loss_fn(o, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0
    for batch in val_dataloader:
        images = batch['images'].to(device)
        findings = batch['findings'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        findings = findings.masked_fill(findings == redacted_id, pad_token_id)
        outputs = model(images, findings, attn_mask)
        vocab_size = outputs.size(-1)
        o = outputs.view(-1, vocab_size)
        target = findings.view(-1)
        loss = loss_fn(o, target)
        val_loss += loss.item()

    val_loss /= len(val_dataloader)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    # wandb.log({"train_loss": train_loss, "val_loss": val_loss})
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
    # wandb.save(f"model_epoch_{epoch + 1}.pth")

torch.save(model.state_dict(), "model_final.pth")
# wandb.save("model_final.pth")
