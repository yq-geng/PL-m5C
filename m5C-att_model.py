import torch
import torch.nn as nn


class RNASeqClassifier(nn.Module):
    
    def __init__(self, num_heads=4, drop_out=0.2, in_channels=4):
        super(RNASeqClassifier, self).__init__()

        # Initialize multi-head attention layers
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)

        # Initialize fully connected layers
        self.fc_layers = self._make_fc_layers(in_channels, drop_out)

    def forward(self, x):
        # [batch_size, input_dim, seq_length]
        # Rearrange dimensions of x for multi-head attention layer input format
        x = x.permute(0, 2, 1)                              # [batch_size, seq_length, input_dim]

        # Pass x through attention layers
        x, _ = self.attention(x, x, x)                      # [batch_size, seq_length, input_dim]

        # Max pooling for x
        x = torch.mean(x, dim=1)                          # [batch_size, input_dim]  

        # Apply fully connected layers (MLP)
        x = self._apply_fc_layers(self.fc_layers, x)        # [batch_size, 1]

        # # Activation function
        # x = torch.sigmoid(x)    

        return x

    def _make_fc_layers(self, in_features, dropout_rate):
        """Create a sequence of fully connected layers where each layer's feature size is half of the previous layer's feature size, starting from in_features/2."""
        layers = nn.ModuleList()
        current_features = in_features // 2  # Start from in_features / 2

        # Loop to create layers where each has half the features of the previous one
        while current_features >= 2:  # Continue until the feature size reduces to below 2
            layers.append(nn.Linear(in_features, current_features))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = current_features  # Update in_features for the next layer
            current_features //= 2  # Halve the number of features for the next layer

        # Add the final layer to reduce to 1 output feature
        layers.append(nn.Linear(in_features, 1))
        return layers

    def _apply_fc_layers(self, layers, x):
        """Apply a sequence of fully connected layers to the input x"""
        for layer in layers:
            x = layer(x)
        return x
    
    
