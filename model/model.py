
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder


class Bert(nn.Module) :

    def __init__(self, config) :
        super().__init__()
        self.config = config
        self.encoder = BertEncoder(config)

        # Position Embedding
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # Album Sequence
        self.album_embed = nn.Embedding(config.album_size, config.hidden_size)
        self.genre_embed = nn.Embedding(config.genre_size, config.hidden_size)
        self.country_embed = nn.Embedding(config.country_size, config.hidden_size)
        
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Dropouts
        self.dropouts = nn.ModuleList([nn.Dropout(config.classifier_dropout) for _ in range(5)])
        
        # Classification
        self.classification_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.01)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


    def forward(
        self, 
        album_input, 
        genre_input, 
        country_input,
    ) :

        mask_token_id = self.config.album_size - 2
        batch_size, seq_size = album_input.shape

        # Album Sequence
        album_tensor = self.album_embed(album_input)
        genre_tensor = self.genre_embed(genre_input)
        country_tensor = self.country_embed(country_input)

        # Attention
        pos_ids = self.position_ids[:, :seq_size]
        pos_tensor = self.position_embeddings(pos_ids)

        # Input Tensor
        input_tensor = album_tensor + genre_tensor + country_tensor + pos_tensor
        input_tensor = self.layernorm(input_tensor)
        input_tensor = self.dropout(input_tensor)
        
        attention_mask = torch.where(album_input==mask_token_id, 1.0, 0.0) * (-1e+20)
        attention_mask = attention_mask.view(batch_size, 1, 1, seq_size)

        # Output Tensor
        sequence_output = self.encoder(
            hidden_states=input_tensor,
            attention_mask=attention_mask
        )[0]

        for i in range(5) :
            if i == 0 :
                logits = self.classification_head(self.dropouts[i](sequence_output))
            else :
                logits += self.classification_head(self.dropouts[i](sequence_output))

        logits /= len(self.dropouts)
        return logits