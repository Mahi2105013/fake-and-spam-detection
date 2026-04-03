import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

META_DIM = 17


class GatedFusionModel(nn.Module):
 

    def __init__(self, roberta_name='roberta-base', num_labels=3,
                 meta_dim=META_DIM, meta_hidden=32, dropout=0.3,
                 class_weights=None):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name)
        hidden = self.roberta.config.hidden_size  # 768

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, meta_hidden),
            nn.GELU(),
        )

        self.gate = nn.Linear(hidden + meta_hidden, hidden)
        self.meta_expand = nn.Linear(meta_hidden, hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fn = nn.CrossEntropyLoss(weight=w)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, metadata, labels=None):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]           # [B, 768]

        meta_emb = self.meta_mlp(metadata)                  # [B, meta_hidden]

        combined = torch.cat([cls_emb, meta_emb], dim=1)    # [B, 768+meta_hidden]
        g = torch.sigmoid(self.gate(combined))               # [B, 768]
        meta_exp = self.meta_expand(meta_emb)                # [B, 768]
        fused = g * cls_emb + (1 - g) * meta_exp            # [B, 768]

        logits = self.classifier(fused)                      # [B, num_labels]

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return type('ModelOutput', (), {'loss': loss, 'logits': logits})()


class TextSCNN(nn.Module):
  

    def __init__(self, vocab_size, embed_dim=128, num_labels=3,
                 sentence_per_review=8, words_per_sentence=16,
                 filter_widths_sent=(3, 4, 5), num_filters_sent=100,
                 filter_widths_doc=(2, 3), num_filters_doc=100,
                 dropout=0.3, class_weights=None):
        super().__init__()

        self.sentence_per_review = sentence_per_review
        self.words_per_sentence = words_per_sentence
        self.max_len = sentence_per_review * words_per_sentence  # 128

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)

        self.sent_convs = nn.ModuleList([
            nn.Conv2d(1, num_filters_sent, kernel_size=(fs, embed_dim))
            for fs in filter_widths_sent
        ])
        sent_out_dim = num_filters_sent * len(filter_widths_sent)

        self.doc_convs = nn.ModuleList([
            nn.Conv2d(1, num_filters_doc, kernel_size=(fs, sent_out_dim))
            for fs in filter_widths_doc
        ])
        doc_out_dim = num_filters_doc * len(filter_widths_doc)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(doc_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fn = nn.CrossEntropyLoss(weight=w)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None, **kwargs):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        if seq_len < self.max_len:
            pad = torch.ones(batch_size, self.max_len - seq_len,
                             dtype=torch.long, device=input_ids.device)
            input_ids = torch.cat([input_ids, pad], dim=1)
        else:
            input_ids = input_ids[:, :self.max_len]

        x = input_ids.view(batch_size, self.sentence_per_review,
                           self.words_per_sentence)

        x = self.embedding(x)
        x = x.view(batch_size * self.sentence_per_review,
                    self.words_per_sentence, -1)
        x = x.unsqueeze(1)

        sent_features = []
        for conv in self.sent_convs:
            h = F.relu(conv(x))
            h = F.max_pool2d(h, kernel_size=(h.size(2), 1))
            sent_features.append(h)
        sent_features = torch.cat(sent_features, dim=1).squeeze(3).squeeze(2)

        sent_features = sent_features.view(batch_size, self.sentence_per_review, -1)
        sent_features = sent_features.unsqueeze(1)

        doc_features = []
        for conv in self.doc_convs:
            h = F.relu(conv(sent_features))
            h = F.max_pool2d(h, kernel_size=(h.size(2), 1))
            doc_features.append(h)
        doc_features = torch.cat(doc_features, dim=1).squeeze(3).squeeze(2)

        logits = self.classifier(doc_features)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return type('ModelOutput', (), {'loss': loss, 'logits': logits})()
