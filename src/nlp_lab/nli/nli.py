
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizerFast

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

DEBUG = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NLIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizerFast, max_length: int = 512) -> None:
        if DEBUG:
            df = df[:100]

        self.tokenizer = tokenizer
        self.max_length = max_length

        df = NLIDataset.clean_df_by_label(df)
        labels = torch.LongTensor(df['label'].values)

        encodings = self.tokenizer(
            df['premise'].values.tolist(),
            df['hypothesis'].values.tolist(),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_token_type_ids=False,
            return_attention_mask=True,
            verbose=True)

        self.input_ids = encodings['input_ids']
        self.attention_masks = encodings['attention_mask']
        self.labels = labels

    @staticmethod
    def clean_df_by_label(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, 'label'] = pd.to_numeric(df.loc[:, 'label'], errors="coerce").astype("int")
        df = df[df[df.columns[-1]].isin([0, 1, 2])]
        return df

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> pd.DataFrame:
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]


class NLIModel(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3) -> None:
        super(NLIModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                labels: torch.Tensor = None
        ) -> torch.Tensor:
        """Docstring.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return outputs


def train_classifier(train_loader: DataLoader, val_loader: DataLoader, epochs: int = 7, lr: float = 1e-5) -> nn.Module:
    """Docstring.

    Returns:
        The trained model.
    """
    model = NLIModel()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    min_val_loss = np.inf

    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for batch in tqdm(train_loader):
            # Fetch data and move to device
            input_ids, attention_mask, label = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            label = label.to(DEVICE)
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(input_ids, attention_mask, None, label)
            loss = output.loss  # BERT automatically calls the CrossEntropyLoss when you pass the labels
            train_loss += loss
            # Backprop
            loss.backward()
            # Gradient descent
            optimizer.step()

        train_loss /= len(train_loader)
        model.eval()

        with torch.no_grad():
            val_loss = 0
            num_correct = 0

            for batch in tqdm(val_loader):
                # Fetch data and move to device
                input_ids, attention_mask, label = batch
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                label = label.to(DEVICE)
                # Predict and calculate metrics
                output = model(input_ids, attention_mask, None, label)
                val_loss += output.loss
                preds = torch.argmax(output.logits, dim=1)
                num_correct += torch.count_nonzero(preds == label)

            val_loss /= len(val_loader)
            val_acc = num_correct / (len(val_loader) * val_loader.batch_size)

            # Save best model
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                torch.save(model.state_dict(), "model.pt")

        # Log training summary every epoch
        print(f"Epoch {epoch+1}: \nTrain Loss = {train_loss:.4f}\nVal Loss = {val_loss:.4f} Val Acc = {val_acc:.4f}")

    return model


if __name__ == "__main__":
    snli = load_dataset("snli")

    train_dataset = NLIDataset(snli["train"].to_pandas(), tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    val_dataset = NLIDataset(snli["validation"].to_pandas(), tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'))
    val_loader = DataLoader(val_dataset, batch_size=32)

    test_dataset = NLIDataset(snli["test"].to_pandas(), tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased'))
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = train_classifier(train_loader, val_loader)
