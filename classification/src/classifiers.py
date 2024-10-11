import math
import torch
import torch.nn as nn

# These x are not sequences. They are just one per video
SINGULAR_FEATURES = ('asr_sentiment', 'ocr_sentiment')


def set_dropout(module, dropout):
    if isinstance(module, nn.Dropout):
        module.p = dropout
    for child_module in module.children():
        set_dropout(child_module, dropout)


class MLP(nn.Module):
    def __init__(self, feature_dims, n_layers, d_layers, d_output, dropout=0.5):
        super().__init__()
        # Takes batch first
        input_size = sum(list(feature_dims.values()))
        self.model = []
        sizes = [input_size] + [d_layers] * (n_layers - 1) + [d_output]
        for i in range(len(sizes) - 1):
            self.model.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                self.model.append(nn.Dropout(dropout, inplace=True))
                self.model.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*self.model)
    def forward(self, x):
        for feature_name in x.keys():
            x[feature_name] = torch.mean(x[feature_name], 1, keepdim=False)
        # Concatenate along channel dimension
        x = torch.cat([x[key] for key in sorted(x.keys())], dim=-1) 
        x = self.model(x)
        return x
    

class Transformer(nn.Module):

    def __init__(self, n_layers=None, n_heads=None, d_model=None, d_ff=None,
                dropout=0.0, learned_position=False, positional_encoding=True,
                max_input_len=1792):
        super(Transformer, self).__init__()
        # Transformer takes seqence first
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, dropout, learned=learned_position, max_len=max_input_len)
        if d_ff == None:
            d_ff = d_model * 4
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_ff, dropout=dropout)
        norm = nn.LayerNorm(d_model)
        self.transformer_body = TransformerEncoder(encoder_layers, n_layers, norm=norm)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        if self.positional_encoding:
            x = self.pos_encoder(x)
        x = self.transformer_body(x, src_mask,
                    src_key_padding_mask=src_key_padding_mask)
        return x



class SingleTransformer(nn.Module):
    def __init__(
        self, feature_lengths, feature_dims, n_layers, n_heads, d_output, dropout, 
        d_model):
        super(SingleTransformer, self).__init__()
        # Takes sequence first

        self.encoders = nn.ModuleDict()
        self.sep_vectors = nn.ParameterDict()
        self.positional_encoding = nn.ModuleDict()

        lengths = list(feature_lengths.values())
        self.total_len = sum(lengths) + len(lengths) + 1   # including <SEP> tokens
        self.d_model = d_model

        d_ff = self.d_model * 4

        for feature, dim in feature_dims.items():
            self.sep_vectors[feature] = nn.Parameter(torch.rand((1, 1, self.d_model), requires_grad=True))
            self.positional_encoding[feature] = PositionalEncoding(self.d_model, learned=True, max_len=self.total_len)

            self.encoders[feature] = nn.Linear(dim, self.d_model)

        self.transformer = Transformer(d_model=self.d_model, n_layers=n_layers, n_heads=n_heads, 
                                       d_ff=d_ff, dropout=dropout,
                                       positional_encoding=False)
        self.cls_vector = nn.Parameter(torch.rand((1, 1, self.d_model), requires_grad=True))
        self.decoder = nn.Linear(self.d_model, d_output)

    def forward(self, x):
        
        for feature_name in x.keys():
            # Convert to sequence-first
            x[feature_name] = x[feature_name].transpose(0, 1)

            # encode to achieve same size with other x
            x[feature_name] = self.encoders[feature_name](x[feature_name])     
            
            # add different <SEP> vector to each feature
            sep_vector = self.sep_vectors[feature_name].expand(-1, x[feature_name].shape[1], -1)
            # print(feature, sep_vector.shape, x[feature].shape)
            x[feature_name] = torch.cat((sep_vector, x[feature_name]), dim=0)
            x[feature_name] = self.positional_encoding[feature_name](x[feature_name])

        # Concatenate in temporal dimension
        x = torch.cat([x[key] for key in sorted(x.keys())], dim=0)     

        # concatenate cls_vector
        cls_vector = self.cls_vector.expand(-1, x.shape[1], -1)
        x = torch.cat((cls_vector, x), dim=0)
        x = self.transformer(x)
        x = x[0, :, :]
        x = self.decoder(x)
        return x
    


class MultiTransformer(nn.Module):
    def __init__(
        self, feature_lengths, feature_dims, n_layers, n_heads, d_output, dropout,
        learned_position=True, d_model=-1):
        super(MultiTransformer, self).__init__()

        self.d_model = d_model

        self.encoders = nn.ModuleDict()
        self.transformers = nn.ModuleDict()
        self.cls_vectors = nn.ParameterDict()

        d_decoder_input = 0
        for feature_name, dim in feature_dims.items():
            input_len = feature_lengths[feature_name] + 1    # +1 for <CLS> token
            if self.d_model > 0:     # use encoder to match
                d = self.d_model
                self.encoders[feature_name] = nn.Linear(dim, d)
            else:
                d = dim

            d_ff = d * 4
            d_decoder_input += d
            
            if feature_name not in SINGULAR_FEATURES:
                self.transformers[feature_name] = Transformer(
                        d_model=d, n_layers=n_layers, n_heads=n_heads, max_input_len=input_len,
                        d_ff=d_ff, dropout=dropout, learned_position=learned_position)

                self.cls_vectors[feature_name] = nn.Parameter(torch.rand((1, 1, d), requires_grad=True))
                
        self.decoder = nn.Linear(d_decoder_input, d_output)

    def forward(self, x):
        for feature_name in x.keys():
            # Convert to sequence-first
            x[feature_name] = x[feature_name].transpose(0, 1)
            if self.d_model > 0:
                x[feature_name] = self.encoders[feature_name](x[feature_name])
            if feature_name not in SINGULAR_FEATURES:
                # concatenate cls_vector
                cls_vector = self.cls_vectors[feature_name].expand(-1, x[feature_name].shape[1], -1)
                x[feature_name] = torch.cat((cls_vector, x[feature_name]), dim=0)
                x[feature_name] = self.transformers[feature_name](x[feature_name])

            x[feature_name] = x[feature_name][0, :, :]

        # Concatenate in feature dimension
        x = torch.cat([x[key] for key in sorted(x.keys())], dim=-1)      

        x = self.decoder(x)
        return x
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, learned=False):
        super(PositionalEncoding, self).__init__()
        # Takes sequence first
        self.dropout = nn.Dropout(p=dropout)
        if learned:
            self.pe = nn.Parameter(torch.empty(max_len, 1, d_model).normal_(std=0.02))
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), ...]
        return self.dropout(x)
    

def init_model(config):
    device = 'cpu' if not torch.cuda.is_available() else config['device']
    if config['model'] == 'single_transformer':
        model = SingleTransformer(
            feature_lengths=config['feature_lengths'],
            feature_dims=config['feature_dims'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_output=config['n_labels'],
            dropout=config['dropout'],
            d_model=config['d_model'],
        )

    elif config['model'] == 'multi_transformer':
        model = MultiTransformer(
            feature_lengths=config['feature_lengths'],
            feature_dims=config['feature_dims'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_output=config['n_labels'],
            dropout=config['dropout'],
            learned_position=not config['fixed_position_encoding'],
        )

    elif config['model'] == 'mlp':
        model = MLP(
            feature_dims=config['feature_dims'],
            n_layers=config['n_layers'],
            d_layers=config['d_model'],
            d_output=config['n_labels'],
            dropout=config['dropout']
        )
        
    model = model.to(device)
    return model


