from transformers.configuration_utils import PretrainedConfig

class BertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        album_size=30000,
        genre_size=30,
        country_size=20,
        age_size=15,
        gender_size=2,
        pr_interest_size=10,
        ch_interest_size=10,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=256,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=256,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        classifier_dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.album_size = album_size
        self.genre_size = genre_size
        self.country_size = country_size
        self.age_size = age_size
        self.gender_size = gender_size
        self.pr_interest_size = pr_interest_size
        self.ch_interest_size = ch_interest_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.classifier_dropout = classifier_dropout