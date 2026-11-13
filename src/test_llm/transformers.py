import torch
import torch.nn as nn
import math

###############################################################################
# Helper Functions and Utilities
###############################################################################

def relu(x):
    """
    ReLU activation: max(0, x)
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output with negative values set to zero
    """
    return torch.where(x > 0, x, torch.zeros_like(x))

def dropout(x, p=0.1, training=True):
    """
    Dropout implementation for regularization:
      - Create a mask of the same shape as x.
      - Drop elements with probability p.
      - Scale the remaining elements by 1 / (1 - p) to maintain expected values.
      
    Args:
        x (torch.Tensor): Input tensor
        p (float): Dropout probability (0.0 to 1.0)
        training (bool): Whether to apply dropout (True) or not (False)
        
    Returns:
        torch.Tensor: Output with dropout applied
    """
    if not training or p <= 0.0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return (mask * x) / (1.0 - p)

def layer_norm(x, eps=1e-5, gamma=None, beta=None):
    """
    Layer Normalization to stabilize training:
      - Normalize across the last dimension (feature dimension).
      - Apply optional learnable scale (gamma) and shift (beta) parameters.
      
    Args:
        x (torch.Tensor): Input tensor
        eps (float): Small constant for numerical stability
        gamma (torch.Tensor, optional): Scale parameter
        beta (torch.Tensor, optional): Shift parameter
        
    Returns:
        torch.Tensor: Normalized tensor with optional scale and shift
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    if gamma is not None:
        x_norm = x_norm * gamma
    if beta is not None:
        x_norm = x_norm + beta
    return x_norm

def softmax(x, dim=-1):
    """
    Custom softmax along a specified dimension.
    Computes a more numerically stable softmax using the max-trick.
    
    Args:
        x (torch.Tensor): Input tensor
        dim (int): Dimension along which to apply softmax
        
    Returns:
        torch.Tensor: Softmax probabilities that sum to 1 along dim
    """
    x_exp = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def matmul3d(a, b):
    """
    A simple batch matrix multiply for 3D tensors:
      - a: [N, seq, d1]
      - b: [N, d1, d2]
      => out: [N, seq, d2]
      
    Args:
        a (torch.Tensor): First tensor of shape [N, seq, d1]
        b (torch.Tensor): Second tensor of shape [N, d1, d2]
        
    Returns:
        torch.Tensor: Output tensor of shape [N, seq, d2]
    """
    return torch.bmm(a, b)

###############################################################################
# "Linear" and "LayerNorm" Classes
###############################################################################

class Linear(nn.Module):
    """
    Linear layer:
    y = xW^T + b
    
    Args:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights using Xavier/Glorot uniform for better gradient flow
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * math.sqrt(2.0 / (in_features + out_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        """
        Forward pass through the linear layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, seq_length, in_features]
                          or [N, in_features]
            
        Returns:
            torch.Tensor: Output tensor of shape [N, seq_length, out_features]
                          or [N, out_features]
        """
        # Handle both 2D and 3D inputs
        if len(x.shape) == 3:
            # Case: [batch_size, seq_length, features]
            N, seq_len, _ = x.shape
            
            # Flatten for matrix multiplication: (N*seq_len, in_features)
            x_2d = x.reshape(-1, self.in_features)
            
            # Apply linear transformation: y = xW^T + b
            out_2d = x_2d.mm(self.weight.t()) + self.bias  # shape: (N*seq_len, out_features)
            
            # Reshape back to original batch and sequence dimensions
            out = out_2d.reshape(N, seq_len, self.out_features)
        else:
            # Case: [batch_size, features] (for classification head)
            # Direct matrix multiplication
            out = x.mm(self.weight.t()) + self.bias
        
        return out

class LayerNorm(nn.Module):
    """
    Layer normalization with learnable parameters:
    - Normalizes input across feature dimension
    - Applies learnable scale (gamma) and shift (beta)
    
    Args:
        dim (int): Feature dimension to normalize
        eps (float): Small constant for numerical stability
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Initialize gamma and beta to 1 and 0 by default
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        """
        Apply layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, seq_length, dim]
            
        Returns:
            torch.Tensor: Normalized tensor of same shape
        """
        return layer_norm(x, eps=self.eps, gamma=self.gamma, beta=self.beta)

###############################################################################
# Positional Encoding
###############################################################################

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding as described in 'Attention Is All You Need'.
    
    Creates sinusoidal embeddings to inject position information:
    - Even dimensions use sine
    - Odd dimensions use cosine
    - Each dimension uses a different frequency
    
    Args:
        embed_size (int): Dimension of the model
        max_seq_length (int): Maximum sequence length to pre-compute
        dropout_p (float): Dropout probability
    """
    def __init__(self, embed_size, max_seq_length=5000, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.max_seq_length = max_seq_length
        
        # Create a buffer of positional encodings up to max_seq_length
        pe = torch.zeros(max_seq_length, embed_size)
        
        # Create a vector of positions: [0, 1, 2, ..., max_seq_length-1]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Create a vector of frequency scalars for each dimension
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        
        # Fill in the positional encoding buffer:
        # - Even indices get sine values
        # - Odd indices get cosine values
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:embed_size // 2])
        
        # Add batch dimension and store for later use
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: [1, max_seq_length, embed_size]
    
    def forward(self, x, training=True):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape [N, seq_length, embed_size]
            training (bool): Whether to apply dropout
            
        Returns:
            torch.Tensor: Embeddings with positional encoding added
        """
        seq_length = x.size(1)
        assert seq_length <= self.max_seq_length, f"Sequence length {seq_length} exceeds maximum {self.max_seq_length}"
        
        # Add the pre-computed positional encodings to the input
        x = x + self.pe[:, :seq_length, :]
        
        # Apply dropout during training
        return dropout(x, p=self.dropout_p, training=training)

###############################################################################
# Multi-Head Attention
###############################################################################

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention implemented as described in 
    'Attention Is All You Need' (Vaswani et al., 2017).
    
    This implements multi-headed self-attention where the input is projected into
    query, key, and value representations. Attention scores are computed between
    queries and keys, then used to create weighted sums of values.
    
    Args:
        embed_size (int): Dimensionality of the input embeddings
        num_heads (int): Number of attention heads to use
    """
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Linear projections for query, key, and value transformations
        # Each transforms the full embedding vector before splitting across heads
        self.query = Linear(embed_size, embed_size)
        self.key = Linear(embed_size, embed_size)
        self.value = Linear(embed_size, embed_size)
        
        # Final linear layer that combines and mixes information across heads
        self.fc_out = Linear(embed_size, embed_size)
    
    def forward(self, x, mask=None):
        """
        Perform the multi-head self-attention operation.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, seq_length, embed_size]
            mask (torch.Tensor, optional): Attention mask of shape 
                                          [N, 1, seq_length, seq_length]
                                          where 0 indicates positions to mask out
        
        Returns:
            torch.Tensor: Output tensor of shape [N, seq_length, embed_size]
        """
        N, seq_length, _ = x.shape

        # Step 1: Project input into query, key, and value representations
        # Each projection applies a learned linear transformation to the input
        Q = self.query.forward(x)  # Shape: [N, seq_length, embed_size]
        K = self.key.forward(x)    # Shape: [N, seq_length, embed_size]
        V = self.value.forward(x)  # Shape: [N, seq_length, embed_size]
        
        # Step 2: Reshape projections to separate the heads dimension
        # This allows each head to attend to different parts of the input
        Q = Q.reshape(N, seq_length, self.num_heads, self.head_dim)  # [N, seq, heads, head_dim]
        K = K.reshape(N, seq_length, self.num_heads, self.head_dim)
        V = V.reshape(N, seq_length, self.num_heads, self.head_dim)
        
        # Step 3: Transpose for attention calculation [N, heads, seq, head_dim]
        # This makes the sequence dimension inner for efficient matrix multiply
        Q = Q.transpose(1, 2)  # (N, num_heads, seq_length, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Step 4: Compute scaled dot-product attention scores
        # Scaling by sqrt(head_dim) helps prevent dot products from growing too large,
        # which would push the softmax into regions with small gradients
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: [N, heads, seq_length, seq_length]
        
        # Step 5: Apply optional mask (for causal attention or padding)
        if mask is not None:
            # Set masked positions to a very negative value (approx. -infinity)
            # This ensures they contribute ~0 after softmax
            attention_scores = attention_scores.masked_fill(mask == 0, float('-1e9'))
        
        # Step 6: Apply softmax to get attention weights
        # Softmax ensures weights sum to 1 for each query position
        attention_weights = softmax(attention_scores, dim=-1)
        # Shape: [N, heads, seq_length, seq_length]
        
        # Step 7: Compute weighted sum of values according to attention weights
        # This implements the core attention mechanism
        out = torch.matmul(attention_weights, V)
        # Shape: [N, heads, seq_length, head_dim]
        
        # Step 8: Transpose and reshape back to original dimensions
        # This combines the output from all attention heads
        out = out.transpose(1, 2).reshape(N, seq_length, self.embed_size)
        # Shape: [N, seq_length, embed_size]
        
        # Step 9: Apply final linear transformation to mix information across heads
        out = self.fc_out.forward(out)
        # Shape: [N, seq_length, embed_size]
        
        return out

###############################################################################
# Position-wise Feed-Forward Network
###############################################################################

class FeedForward(nn.Module):
    """
    Two-layer feed-forward network as described in 
    'Attention Is All You Need' (Vaswani et al., 2017).
    
    Architecture: Linear -> ReLU -> Linear
    
    The inner dimension is typically 4x the model dimension.
    
    Args:
        embed_size (int): Model dimension
        ff_hidden_mult (int): Multiplier for hidden dimension (typically 4)
    """
    def __init__(self, embed_size, ff_hidden_mult=4):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = embed_size * ff_hidden_mult
        
        # First linear expands to larger hidden size
        self.fc1 = Linear(embed_size, self.hidden_size)
        # Second linear projects back to model dimension
        self.fc2 = Linear(self.hidden_size, embed_size)
    
    def forward(self, x):
        """
        Apply the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, seq_length, embed_size]
            
        Returns:
            torch.Tensor: Output tensor of same shape
        """
        # Expand to larger hidden size
        x = self.fc1.forward(x)
        # Apply ReLU activation
        x = relu(x)
        # Project back to model dimension
        x = self.fc2.forward(x)
        return x

###############################################################################
# Transformer Block
###############################################################################

class TransformerBlock(nn.Module):
    """
    Single Transformer Block that combines:
      - Multi-Head Self-Attention
      - Feed-Forward Network
      - Residual connections
      - Layer normalization
      - Dropout for regularization
      
    This implements the core building block of the Transformer architecture
    as described in 'Attention Is All You Need' (Vaswani et al., 2017).
    
    Args:
        embed_size (int): Model dimension
        num_heads (int): Number of attention heads
        dropout_p (float): Dropout probability
        ff_hidden_mult (int): Multiplier for feed-forward hidden dimension
    """
    def __init__(self, embed_size, num_heads, dropout_p=0.1, ff_hidden_mult=4):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_hidden_mult)
        self.dropout_p = dropout_p
    
    def forward(self, x, mask=None, training=True):
        """
        Apply the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, seq_length, embed_size]
            mask (torch.Tensor, optional): Attention mask
            training (bool): Whether to apply dropout
            
        Returns:
            torch.Tensor: Output tensor of same shape
        """
        # Sub-block 1: Multi-Head Attention with residual connection and layer norm
        attn_out = self.attention.forward(x, mask=mask)
        x = x + dropout(attn_out, p=self.dropout_p, training=training)  # Residual connection
        x = self.norm1.forward(x)  # Layer normalization
        
        # Sub-block 2: Feed-Forward with residual connection and layer norm
        ff_out = self.ff.forward(x)
        x = x + dropout(ff_out, p=self.dropout_p, training=training)  # Residual connection
        x = self.norm2.forward(x)  # Layer normalization
        
        return x

###############################################################################
# Transformer Encoder
###############################################################################

class TransformerEncoder(nn.Module):
    """
    Full transformer encoder stack consisting of:
      - Input embeddings
      - Positional encoding
      - Multiple transformer blocks
      
    This implements the encoder portion of the Transformer architecture
    as described in 'Attention Is All You Need' (Vaswani et al., 2017).
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_size (int): Model dimension
        num_layers (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        max_seq_length (int): Maximum sequence length
        dropout_p (float): Dropout probability
        ff_hidden_mult (int): Multiplier for feed-forward hidden dimension
    """
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, 
                 max_seq_length=5000, dropout_p=0.1, ff_hidden_mult=4):
        super().__init__()
        # Token embedding layer: converts token indices to vectors
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_size)
        
        # Scale embeddings to match model dimension for better training dynamics
        self.embed_scale = math.sqrt(embed_size)
        
        # Positional encoding to inject position information
        self.pos_encoding = PositionalEncoding(embed_size, max_seq_length, dropout_p)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout_p, ff_hidden_mult)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization (after the last transformer block)
        self.norm = LayerNorm(embed_size)
        
        self.dropout_p = dropout_p
        self.embed_size = embed_size
    
    def forward(self, x, mask=None, training=True):
        """
        Apply the transformer encoder.
        
        Args:
            x (torch.Tensor): Input token indices of shape [N, seq_length]
            mask (torch.Tensor, optional): Attention mask
            training (bool): Whether to apply dropout
            
        Returns:
            torch.Tensor: Output tensor of shape [N, seq_length, embed_size]
        """
        # Make sure x is a LongTensor for embedding lookup
        x = x.long()
        
        # Convert token indices to embeddings and scale
        embedded = self.token_embedding(x) * self.embed_scale
        
        # Add positional encoding
        x = self.pos_encoding.forward(embedded, training=training)
        
        # Apply each transformer block in sequence
        for layer in self.layers:
            x = layer.forward(x, mask=mask, training=training)
        
        # Apply final layer normalization
        x = self.norm.forward(x)
        
        return x

###############################################################################
# Complete Transformer Model (Encoder-Only)
###############################################################################

class Transformer(nn.Module):
    """
    Encoder-only Transformer model for sequence classification or other tasks.
    
    This implements a complete transformer model that can be used for
    various NLP tasks like text classification, sequence tagging, etc.
    
    Args:
        vocab_size (int): Size of vocabulary
        embed_size (int): Model dimension
        num_layers (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        num_classes (int): Number of output classes (for classification)
        max_seq_length (int): Maximum sequence length
        dropout_p (float): Dropout probability
        ff_hidden_mult (int): Multiplier for feed-forward hidden dimension
    """
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, num_classes,
                 max_seq_length=5000, dropout_p=0.1, ff_hidden_mult=4):
        super().__init__()
        # Encoder stack
        self.encoder = TransformerEncoder(
            vocab_size, embed_size, num_layers, num_heads,
            max_seq_length, dropout_p, ff_hidden_mult
        )
        
        # Classification head: projects from embed_size to num_classes
        self.classifier = Linear(embed_size, num_classes)
        
        self.dropout_p = dropout_p
    
    def forward(self, x, mask=None, training=None):
        """
        Apply the complete transformer model.
        
        For classification, typically use the output corresponding to a special
        token (like [CLS] in BERT) or pool across the sequence dimension.
        
        Args:
            x (torch.Tensor): Input token indices of shape [N, seq_length]
            mask (torch.Tensor, optional): Attention mask
            training (bool): Whether to apply dropout
            
        Returns:
            torch.Tensor: Output tensor of shape [N, num_classes]
        """
        if training is None:
            training = self.training

        # Apply encoder stack
        encoder_output = self.encoder.forward(x, mask=mask, training=training)
        
        # Use first token ([CLS]) for classification, like in BERT
        # Alternative: pooling across sequence dimension
        first_token = encoder_output[:, 0, :]
        
        # Apply dropout and classification head
        pooled_output = dropout(first_token, p=self.dropout_p, training=training)
        logits = self.classifier.forward(pooled_output)
        
        return logits

###############################################################################
# Example Usage / Demo
###############################################################################

def create_padding_mask(seq, pad_idx=0):
    """
    Create attention mask for padding tokens.
    
    Args:
        seq (torch.Tensor): Input tensor of shape [N, seq_length]
        pad_idx (int): Index used for padding
        
    Returns:
        torch.Tensor: Mask tensor of shape [N, 1, 1, seq_length]
        where 0 indicates positions to mask out
    """
    # Create mask for padding tokens (1 for valid tokens, 0 for padding)
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_causal_mask(seq_length):
    """
    Create causal (look-ahead) mask for decoder self-attention.
    
    Args:
        seq_length (int): Length of sequence
        
    Returns:
        torch.Tensor: Lower triangular mask of shape [1, 1, seq_length, seq_length]
        where 0 indicates positions to mask out
    """
    # Create lower triangular mask (1s in lower triangle, 0s elsewhere)
    mask = torch.tril(torch.ones(1, 1, seq_length, seq_length))
    return mask

def example():
    # Hyperparameters
    vocab_size = 10000
    embed_size = 64
    num_layers = 3
    num_heads = 8
    num_classes = 2  # Binary classification for demonstration
    max_seq_length = 100
    ff_hidden_mult = 4
    dropout_p = 0.1
    
    # Sample input (token indices)
    batch_size, seq_length = 2, 10
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Padding mask for attention
    mask = create_padding_mask(x)
    
    print("Input shape:", x.shape)

    # 1) Create transformer model
    model = Transformer(
        vocab_size, embed_size, num_layers, num_heads, num_classes,
        max_seq_length, dropout_p, ff_hidden_mult
    )
    
    # 2) Forward pass
    output = model.forward(x, mask=mask, training=True)
    
    print("Transformer output shape:", output.shape)  # Should be [2, 2] for binary classification
    
    # 3) Example training loop (just for demonstration)
    print("\nExample of a forward and backward pass (not actually training):")
    
    # Random target labels for binary classification
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Forward pass
    logits = model.forward(x, mask=mask, training=True)
    print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
    
    # Loss computation (not actually performing backward pass)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(f"Loss value: {loss.item()}")
    
    print("\nTransformer model architecture summary:")
    print(f"- Vocab size: {vocab_size}")
    print(f"- Embedding dimension: {embed_size}")
    print(f"- Number of layers: {num_layers}")
    print(f"- Number of attention heads: {num_heads}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Maximum sequence length: {max_seq_length}")
    print(f"- Dropout probability: {dropout_p}")
    print(f"- Feed-forward hidden multiplier: {ff_hidden_mult}")

if __name__ == "__main__":
    example()
