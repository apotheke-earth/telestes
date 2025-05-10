import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        heads: int
    ):
        """
        Initialize the self-attention mechanism.
        """
        super(SelfAttention, self).__init__()

        self.embed_dims = embed_dims
        self.heads = heads

        self.head_dim = embed_dims//heads

        assert (self.head_dim*heads==embed_dims), f"heads={heads} must divide embed size {embed_dims}"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries= nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.output = nn.Linear(self.embed_dims, self.embed_dims)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask=None
    ) -> torch.Tensor:
        """
        Forward pass for the attention mechanism
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
        
        out = self.output(
            out.reshape(
                N, query_len, self.heads*self.head_dim                  
            )
        )
        return out

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        heads: int = 8,
        dropout: float = 0,
        forward_expansion: int = 1,
        activation_function: nn.Module = nn.ReLU
    ):
        """
        Initialize a transformer block,

        We use multi-headed self-attention and feed-forward networks,
        as per the paper
        """
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_dims, heads)
        self.normalizer = nn.LayerNorm(embed_dims)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dims, forward_expansion*embed_dims),
            activation_function(),
            nn.Linear(forward_expansion*embed_dims, embed_dims)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask=None
    ) -> torch.Tensor:
        """
        Forward pass through the entire block, as per the paper
        """
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.normalizer(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.normalizer(forward+x))
        return out

class AttentionGate(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        heads: int = 4
    ):
        """
        Initialize the attention gate

        The gate reduces the dims of the transformer output
        """
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
            token_embeddings
        )

        weights = self.aggregator(out)
        weighted_sum = (weights*token_embeddings).sum(dim=1)
        return weighted_sum
        
    
