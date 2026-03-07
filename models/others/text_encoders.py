from transformers import CLIPTokenizer, CLIPTextModel, BertTokenizer, BertModel
import torch.nn as nn


class CLIPTextEncoderWrapper(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.context_dim = self.text_encoder.config.hidden_size
        
        if freeze:
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False


    def encode_text(self, captions, device=None):
        inputs = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            self.text_encoder.to(device)

        outputs = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        return outputs.last_hidden_state 


class BERTTextEncoderWrapper(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze=True, max_length=77):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.text_encoder = BertModel.from_pretrained(model_name)
        self.max_length = max_length
        self.context_dim = self.text_encoder.config.hidden_size

        if freeze:
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def encode_text(self, captions, device=None):
        inputs = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            self.text_encoder.to(device)

        outputs = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        return outputs.last_hidden_state

    def forward(self, captions, device=None):
        return self.encode_text(captions, device=device)