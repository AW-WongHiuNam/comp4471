import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        pe = torch.zeros(1, max_len, embed_dim)
        
        # *****START OF YOUR CODE*****
        # Correctly creating the div_term using log/exp for better precision
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # *****END OF YOUR CODE*****

        self.register_buffer('pe', pe)

    def forward(self, x):
        N, S, D = x.shape
        output = torch.empty((N, S, D))
        # *****START OF YOUR CODE*****
        output = x + self.pe[:, :S]
        output = self.dropout(output)
        # *****END OF YOUR CODE*****
        return output

class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                # 1. Linear projections for Query, Key, and Value
        # Shapes: (N, S, E), (N, T, E), (N, T, E)
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        # 2. Reshape and transpose for multi-head attention
        # Split embedding dimension E into num_heads (H) and head_dim (E/H)
        # Transpose to bring H to the second dimension for batched matrix multiplication
        # New Shapes: (N, H, S, head_dim), (N, H, T, head_dim), (N, H, T, head_dim)
        q = q.view(N, S, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(N, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Compute Scaled Dot-Product Attention Scores
        # Matrix multiply Q and K^T: (N, H, S, head_dim) @ (N, H, head_dim, T) -> (N, H, S, T)
        scores = torch.matmul(q, k.transpose(-2, -1))
        # Scale by sqrt(head_dim)
        scores = scores / (self.head_dim ** 0.5)

        # 4. Apply Mask
        if attn_mask is not None:
            # attn_mask shape (S, T) broadcasts to (N, H, S, T)
            # mask[i,j] == 0 means ignore, so set score to -infinity
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        # 5. Softmax and Dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # 6. Apply attention weights to Value
        # (N, H, S, T) @ (N, H, T, head_dim) -> (N, H, S, head_dim)
        context = torch.matmul(attn_weights, v)

        # 7. Recombine heads
        # Transpose back: (N, H, S, head_dim) -> (N, S, H, head_dim)
        # Flatten last two dims: (N, S, E)
        context = context.transpose(1, 2).contiguous().view(N, S, E)

        # 8. Output Linear Projection
        output = self.proj(context)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output