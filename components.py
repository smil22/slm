import torch

def token_encoder_v1(vocabulary_units, string):
    """"This function maps the vocabulary units to integer values corresponding to the index of the unit in the vocabulary to encode a string."""
    stoi = {ch:i for i,ch in enumerate(vocabulary_units)}
    return [stoi[char] for char in string]

def token_decoder_v1(vocabulary_units, integers_list):
    """"This function maps the vocabulary units to integer values corresponding to the index of the unit in the vocabulary to decode a list of integers."""
    itos = {i:ch for i,ch in enumerate(vocabulary_units)}
    return ''.join([itos[integer] for integer in integers_list])

class AttentionHead(torch.nn.Module):
    """ One head of self-attention """
    
    def __init__(self,head_size,embedding_dimension,block_size,dropout): # head_size = embedding_dimension // attention_head_number (window size)
        super().__init__()
        self.key = torch.nn.Linear(embedding_dimension, head_size,bias=False)
        self.query = torch.nn.Linear(embedding_dimension, head_size,bias=False)
        self.value = torch.nn.Linear(embedding_dimension, head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size))) # defines a lower triangular matrix mask to apply to the weights
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Attention scores computing ("affinities")
        weights = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        weights = weights.masked_fill(self.tril[:T,:T] == 0,float('-inf')) # (B,T,T)
        weights = torch.nn.functional.softmax(weights,dim=-1) # (B,T,T)
        weights = self.dropout(weights)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = weights @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    
class MultiHeadAttention(torch.nn.Module):
    """ Mutilple head of self-attention in parallel """
    
    def __init__(self,attention_head_number,attention_head_size,embedding_dimension,dropout,block_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([AttentionHead(attention_head_size,embedding_dimension,block_size,dropout) for _ in range(attention_head_number)])
        self.projection = torch.nn.Linear(embedding_dimension,embedding_dimension)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self,x):
        # Apply each head to the input
        out = torch.cat([h(x) for h in self.heads],dim=-1) # (B,T,attention_head_number*attention_head_size) # heads concatenation
        out = self.dropout(self.projection(out))
        return out
    
class FeedForward(torch.nn.Module):
    """" A simple linear layer followed by a non-linearity """
    
    def __init__(self,embedding_dimension,dropout):
        super().__init__()
        output_scale = 512
        self.network = torch.nn.Sequential(
            torch.nn.Linear(embedding_dimension,output_scale*embedding_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(output_scale*embedding_dimension,embedding_dimension), # projection layer
            torch.nn.Dropout(dropout), # dropout layer
        )
        
    def forward(self,x):
        return self.network(x)
    
class TransformerBlock(torch.nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self,attention_head_number,attention_head_size,embedding_dimension,dropout,block_size):
        super().__init__()
        attention_head_size = embedding_dimension // attention_head_number
        self.sa = MultiHeadAttention(attention_head_number,attention_head_size,embedding_dimension,dropout,block_size)
        self.ffwd = FeedForward(embedding_dimension,dropout)
        self.ln1 = torch.nn.LayerNorm(embedding_dimension)
        self.ln2 = torch.nn.LayerNorm(embedding_dimension)
        
    def forward(self,x):
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffwd(self.ln2(x))
        return x
    
class SLModel(torch.nn.Module):
    
    def __init__(self,vocabulary_size,embedding_dimension,block_size,attention_head_number,decoder_block_number,attention_head_size,dropout):
        super().__init__()
        # Each token reads the logits for the next from a lookup table
        self.token_embedding_table = torch.nn.Embedding(vocabulary_size,embedding_dimension)
        self.position_embedding_table = torch.nn.Embedding(block_size,embedding_dimension) 
        self.blocks = torch.nn.Sequential(*[TransformerBlock(attention_head_number,attention_head_size,embedding_dimension,dropout,block_size) for _ in range(decoder_block_number)])
        """ self.blocks = torch.nn.Sequential(
            Block(embedding_dimension,attention_head_number=4),
            Block(embedding_dimension,attention_head_number=4),
            Block(embedding_dimension,attention_head_number=4),
            torch.nn.LayerNorm(embedding_dimension),
        ) """
        self.ln_f = torch.nn.LayerNorm(embedding_dimension) # final layer norm
        self.lm_head = torch.nn.Linear(embedding_dimension,vocabulary_size) # to pass from tokens embedding to logits
        
    def forward(self,index,targets=None):
        B,T = index.shape
        # index and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (Batch,Time,Channel)
        pos_emb = self.position_embedding_table(torch.arange(T,device=index.device)) # (Time,Channel)
        x = tok_emb + pos_emb # (Batch,Time,Channel)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (Batch,Time,vocabulary_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits,targets)
        
        return logits,loss
    
    def generate(self,index,block_size,max_new_tokens=500):
        # idx size is (B,T) and it contains indices in the current context
        for _ in range(max_new_tokens):
            # crop index to the last block_size tokens
            index_cond = index[:,-block_size:]
            # get the predictions
            logits,loss = self(index_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # becomes (B,C)
            # softmax applying to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            index_next = torch.multinomial(probs,num_samples=1) #(B,1)
            # append sampled index to the running sequence
            index = torch.cat((index,index_next), dim=1) # (B,T+1)
        return index