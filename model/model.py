
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder


class Bert(nn.Module) :

    def __init__(self, config) :
        super().__init__()
        self.config = config
        self.encoder = BertEncoder(config)

        self.position_embed = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
            size=(config.max_length, config.hidden_size)), 
            requires_grad=True
        )

        self.age_embed = nn.Embedding(config.age_size, config.hidden_size)
        self.gender_embed = nn.Embedding(config.gender_size, config.hidden_size)

        self.album_embed = nn.Embedding(config.album_size, config.hidden_size)
        self.genre_embed = nn.Embedding(config.genre_size, config.hidden_size)
        self.country_embed = nn.Embedding(config.country_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classification_head = nn.Linear(config.hidden_size, config.num_labels)
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
        age_input,
        gender_input,
    ) :

        mask_token_id = self.config.album_size - 2
        batch_size, seq_size = album_input.shape

        age_tensor = self.age_embed(age_input)
        gender_tensor = self.gender_embed(gender_input)

        profile_tensor = age_tensor + gender_tensor
        profile_tensor = profile_tensor.unsqueeze(1)

        album_tensor = self.album_embed(album_input)
        genre_tensor = self.genre_embed(genre_input)
        country_tensor = self.country_embed(country_input)

        pos_tensor = self.position_embed[:seq_size, :]
        attention_mask = torch.where(album_input==mask_token_id, 1.0, 0.0) * (-1e+20)
        attention_mask = attention_mask.view(batch_size, 1, 1, seq_size)

        input_tensor = album_tensor + genre_tensor + country_tensor + pos_tensor 
        input_tensor = self.LayerNorm(input_tensor)
        input_tensor = self.dropout(input_tensor)

        sequence_output = self.encoder(
            hidden_states=input_tensor + profile_tensor, 
            attention_mask=attention_mask
        )[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classification_head(sequence_output)
        return logits
