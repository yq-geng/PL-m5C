import torch
import torch.nn as nn

# 1D Convolution Block
class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.prelu(x)
        return x

class RNASeqClassifier(nn.Module):
    
    def __init__(self, num_layers=3, num_heads=4, drop_out=0.2, in_channels=4):
        super(RNASeqClassifier, self).__init__()
        base = num_heads * 8 
        conv_output_channels = [base * (2 ** i) for i in range(num_layers)]
        kernel_sizes = 3

        # Initialize 1D convolution layers for processing x1, x2
        self.conv_layers = self._make_conv_layers(num_layers, in_channels, conv_output_channels, kernel_sizes)

        # Initialize fully connected layers
        self.fc_layers = self._make_fc_layers(conv_output_channels[-1], drop_out)

    def forward(self, x):
        # [batch_size, input_dim, seq_length]
        # Pass x through 1D CNN
        x = self._apply_conv_layers(self.conv_layers, x)    # [batch_size, conv_output_dim, seq_length]

        # Rearrange dimensions of x for multi-head attention layer input format
        x = x.permute(0, 2, 1)                              # [batch_size, seq_length, conv_output_dim]

        # Max pooling for x
        x, _ = torch.max(x, dim=1)                          # [batch_size, conv_output_dim]  

        # Apply fully connected layers (MLP)
        x = self._apply_fc_layers(self.fc_layers, x)        # [batch_size, 1]

        # # Activation function
        # x = torch.sigmoid(x)    

        return x

    def _make_conv_layers(self, num_layers, in_channels, out_channels_list, kernel_sizes):
        """Create a sequence of convolution layers"""
        layers = nn.ModuleList()
        for i in range(num_layers):
            layers.append(ConvBlock(in_channels, out_channels_list[i], kernel_sizes))
            in_channels = out_channels_list[i]
        return layers

    def _apply_conv_layers(self, layers, x):
        """Apply a sequence of convolution layers to the input x"""
        for layer in layers:
            x = layer(x)
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
    
    
