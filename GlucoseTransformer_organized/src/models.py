import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for transformer models.
    
    Adds positional information to input embeddings to maintain sequence order information
    since attention mechanisms are inherently permutation invariant.
    
    Args:
        d_model: Dimensionality of the model embeddings
        max_len: Maximum sequence length to pre-compute encodings for
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # [1, max_len, d_model]
        
    def forward(self, x):
        """
        Add positional encodings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encodings added
        """
        return x + self.encoding[:, :x.size(1), :].to(x.device)


class TransformerEncoder_version2(nn.Module):
    """
    Transformer encoder with customized architecture for glucose prediction.
    
    This version uses a different output architecture compared to TransformerEncoder.
    
    Args:
        past_seq_len: Length of input sequence
        num_layers: Number of transformer encoder layers
        d_model: Dimension of the model
        nhead: Number of attention heads
        input_dim: Dimension of input features
        dropout: Dropout probability
    """
    def __init__(self, past_seq_len, num_layers, d_model, nhead, input_dim=1, dropout=0.1):
        super(TransformerEncoder_version2, self).__init__()
        self.d_model = d_model
        self.past_seq_len = past_seq_len
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Initial projection of input data
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Stack of Encoder Layers
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # Multi-Head Attention block
                'attention': nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                ),
                # Add & Norm after attention
                'norm1': nn.LayerNorm(d_model),
                
                # Feed Forward block
                'feed_forward': nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                ),
                # Add & Norm after feed forward
                'norm2': nn.LayerNorm(d_model)
            })
            self.encoder_layers.append(layer)
        
        # Output layers
        self.linear2 = nn.Linear(d_model, 1)
        self.linear3 = nn.Linear(past_seq_len, 1)
        
    def forward(self, src):
        """
        Forward pass through the transformer encoder.
        
        Args:
            src: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor with predictions
        """
        # Initial projection and positional encoding
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        
        # Process through encoder layers
        for layer in self.encoder_layers:
            # Multi-Head Attention
            attn_output, _ = layer['attention'](x, x, x)
            # Add & Norm (first residual connection)
            x = layer['norm1'](x + attn_output)
            # Feed Forward
            ff_output = layer['feed_forward'](x)
            # Add & Norm (second residual connection)
            x = layer['norm2'](x + ff_output)
        
        x = self.linear2(x)
        x = x.squeeze(-1)
        output = self.linear3(x)
        return output
