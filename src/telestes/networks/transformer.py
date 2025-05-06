import torch
import torch.nn as nn
import torch.optim as optim

class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        heads: int
    ):
        """
        Initialize the self-attention mechanism.

        Parameters:
        embed_size (int): The size of the input embeddings.
        heads (int): The number of attention heads in the multi-head attention mechanism.
        """
        super(SelfAttention, self).__init__()

        self.embed_dims = embed_dims
        self.heads = heads
        self.head_dim = embed_dims // heads

        assert (self.head_dim * heads == embed_dims), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.embed_dims, self.embed_dims)

    def forward(self, values, keys, query, mask):
        """
        Perform the forward pass of the self-attention mechanism.

        Parameters:
        values (Tensor): The values used in attention calculation.
        keys (Tensor): The keys used in attention calculation.
        query (Tensor): The query used in attention calculation.
        mask (Tensor, optional): A mask to prevent attention to certain positions (default: None).

        Returns:
        Tensor: The output of the attention mechanism after applying the linear transformation.
        """
        N = query.shape[0]

        value_len = values.shape[1]
        key_len = keys.shape[1]
        query_len = query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        # say the magic words: "nqhd,nkhd->nhqk"
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))
        attention = torch.softmax(energy/(self.embed_dims**(1/2)), dim=3)
        # now the spell becomes: "nhql,nlhd->nqhd"
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        # i say these are magic since traditionally enchanments follow a pattern:
        # the caster of the spell says the magic words (or keywords)
        # and there is an observable effect that happens
        # exactly because of these words.
        # using this definition, one could claim that all programming is magic
        # since we "say" or "write" the magic words in a "spellbook"
        # and then observe the impact of our words.
        # nevertheless i like the idea of only passing strings as args counting

        out = self.fc_out(out.reshape(
            N, query_len, self.heads*self.head_dim
        ))
        return out

        
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        heads: int = 8,
        dropout: float = 0,
        forward_expansion: int = 4
    ):
        """
        Initialize the transformer block, which includes multi-head self-attention 
        and a feed-forward neural network.

        Parameters:
        embed_size (int): The size of the input embeddings.
        heads (int): The number of attention heads for multi-head attention.
        dropout (float): The dropout rate for regularization.
        forward_expansion (int): The expansion factor for the feed-forward network.
        """
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_dims, heads)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dims, forward_expansion*embed_dims),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_dims, embed_dims)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """
        Perform the forward pass of the transformer block.

        Parameters:
        value (Tensor): The values used in attention calculation.
        key (Tensor): The keys used in attention calculation.
        query (Tensor): The query used in attention calculation.
        mask (Tensor, optional): A mask to prevent attention to certain positions (default: None).

        Returns:
        Tensor: The output of the transformer block after attention and feed-forward network.
        """
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        return out

class AttentionGate(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        heads: int = 4
    ):
        super(AttentionGate, self).__init__()

        self.attention = SelfAttention(embed_dims, heads)
        self.aggregator = nn.Sequential(
            nn.Linear(embed_dims, 1),
            nn.Softmax(-1)
        )

    def forward(self, token_embeddings):
        out = self.attention(
            token_embeddings,
            token_embeddings,
            token_embeddings,
            mask = None
        )

        weights = self.aggregator(out)

        return (weights*token_embeddings).sum(dim=1)
        
