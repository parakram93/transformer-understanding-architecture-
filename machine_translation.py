import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import re
import random
from collections import Counter

data = load_dataset("Helsinki-NLP/opus-100", "en-es")

train_data = data["train"].select(range(20000))
train_texts = [item["translation"]["en"] for item in train_data]
train_labels = [item["translation"]["es"] for item in train_data]

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

tokenized_en = [tokenize(t) for t in train_texts]
tokenized_fr = [tokenize(t) for t in train_labels]

def tok(text):
    return [t for sentence in text for t in sentence]

tok_en = tok(tokenized_en)
tok_fr = tok(tokenized_fr)

tgt_vocab = {"<pad>" : 0, "<unk>":1, "<sos>":2, "<eos>":3}
src_vocab = {"<pad>" : 0, "<unk>":1, "<sos>":2, "<eos>":3}

pad_idx = 0


for t in Counter(tok_en).keys():
    if t not in src_vocab:
        src_vocab[t] = len(src_vocab)
for t in Counter(tok_fr).keys():
    if t not in tgt_vocab:
        tgt_vocab[t] = len(tgt_vocab)



max_len = 60

def encode_src(tokens):
    tokens = tokens[:max_len]
    ids =  [src_vocab.get(t, src_vocab["<unk>"]) for t in tokens] 
    return torch.tensor(ids, dtype = torch.long)

def encode_tgt(tokens):
    tokens = tokens[:max_len-2]
    ids = [tgt_vocab["<sos>"]] + [tgt_vocab.get(t, tgt_vocab["<unk>"]) for t in tokens] + [tgt_vocab["<eos>"]]
    return torch.tensor(ids)

encoded_en = [encode_src(x) for x in tokenized_en]
encoded_fr = [encode_tgt(x) for x in tokenized_fr]
print(encoded_fr[0])

class ConversionDataset(Dataset):
    
    def __init__(self, src_list, tgt_list):
        self.src_list = src_list
        self.tgt_list = tgt_list  
    def __len__(self):
        return len(self.src_list)
    
    def __getitem__(self,idx):
        src = self.src_list[idx]
        src_len = len(src)
        
        tgt = self.tgt_list[idx]
        
        tgt_len = len(tgt)
        
        tgt_input = tgt[:-1]
        
        tgt_output = tgt[1:]
        
        return {
            "src" : src,
            "tgt_input":tgt_input,
            "tgt_output":tgt_output,
            "src_len": src_len,
            "tgt_len":tgt_len
        }
        
def transformer_collate(batch):
    
    src_list = [item["src"] for item in batch]
    tgt_input_list = [item["tgt_input"] for item in batch]
    tgt_output_list = [item['tgt_output'] for item in batch]
    src_len_list = [item['src_len'] for item in batch]
    tgt_len_list = [item['tgt_len'] for item in batch]
    
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_input_padded = pad_sequence(tgt_input_list, batch_first=True, padding_value=pad_idx)
    tgt_output_padded = pad_sequence(tgt_output_list, batch_first=True, padding_value=pad_idx)
    
    return{
        "src": src_padded,
        "tgt_input":tgt_input_padded,
        "tgt_output":tgt_output_padded,
        "src_len": torch.tensor(src_len_list)
        ,"tgt_len":torch.tensor(tgt_len_list)
    }

dataset = ConversionDataset(encoded_en, encoded_fr)

dataloader = DataLoader(dataset=dataset, batch_size=32,shuffle=True, collate_fn=transformer_collate)

import torch
import torch.nn as nn
import math

#return (batch,seq_len, d_model
# )
class inputEmbeddings(nn.Module):
    
    def __init__(self, d_model:int, vocab_size: int ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x)* math.sqrt(self.d_model)  #We multiply the embeddings by math.sqrt(self.d_model) to scale the weights so they don't get "washed out" by the Positional Encodings that are added immediately afterward.
    

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
    #create a matrix of shape(seq_len, d_model)
    
        pe = torch.zeros(seq_len,d_model)
         # create a vector of shape (seq_len,1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model)) #torch.arange(0, d_model, 2) . creates a 1-D tensor starting at 0, stopping before d_model, stepping by 2, 

        #apply sin to even positions
        
        pe[:, 0::2] = torch.sin(position* div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        
        x = x + (self.pe[: , :x.shape[1], : ]).requires_grad_(False) #this makes positional embedding of shape equal to shape of x by making seq+len equal. we slice pe to match the shape
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x-mean) / (std + self.eps)  + self.bias
        
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout : float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) #w1 anf b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #w2 and b2
          
        
    def forward(self,x):
        #(batch, seq_len, d_model) ---> (batch, seq_len, d_ff) ---> (batch, seq_len, d_model)
        
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
      
class MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout_rate = dropout
        assert d_model % h == 0, "d_model should be divisible by h"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len_q, d_k) @ (batch, h, d_k, seq_len_k) --> (batch, h, seq_len_q, seq_len_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Use a very large negative number so softmax turns it into 0
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len_q, seq_len_k) @ (batch, h, seq_len_k, d_k) --> (batch, h, seq_len_q, d_k)
        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask=None):
        # Get batch sizes and sequence lengths separately for cross-attention
        B, seq_len_q, _ = q.shape
        B, seq_len_k, _ = k.shape
        B, seq_len_v, _ = v.shape

        # Prepare mask to be broadcastable to (B, h, seq_len_q, seq_len_k)
        if mask is not None:
            if mask.dim() == 2:  # (B, seq_len_k)
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,seq_len_k)
            elif mask.dim() == 3:  # (B, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)  # (B,1,seq_len_q,seq_len_k)
            # now mask is broadcastable to (B, h, seq_len_q, seq_len_k)

        # Linear projections
        query = self.w_q(q).view(B, seq_len_q, self.h, self.d_k).transpose(1, 2)  # (B,h,seq_len_q,d_k)
        key   = self.w_k(k).view(B, seq_len_k, self.h, self.d_k).transpose(1, 2)  # (B,h,seq_len_k,d_k)
        value = self.w_v(v).view(B, seq_len_v, self.h, self.d_k).transpose(1, 2)  # (B,h,seq_len_v,d_k)

        # Scaled dot-product attention
        x, self.attention_scores = MultiheadAttention.attention(query, key, value, mask, self.dropout)

        # .contiguous(): When you transpose a tensor, PyTorch doesn't actually move the data around in your RAM;
        # it just changes the "metadata" (the way it reads the memory).
        # Why it's here: Many operations, including .view(), require the data to be stored in a single, continuous block of memory.
        # .contiguous() physically rearranges the data in RAM to match the new transposed order.
        # Without this, the next step (.view) would throw a runtime error.
        x = x.transpose(1, 2).contiguous()  # (B, seq_len_q, h, d_k)
        B_, seq_len_, h_, d_k_ = x.shape
        x = x.view(B_, seq_len_, h_ * d_k_)  # (B, seq_len_q, d_model), safe: compute dims from tensor

        # We do this to calculate weighted sum of different heads because, at first different heads only exist,
        # but are not talking together. After fully connected layer is applied, they are talking to each other.
        return self.w_o(x)

         
class Residual_connection(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))   
          
class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiheadAttention, feed_forward_block: FeedForward, dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward_block
        
        #nn.ModuleList is a PyTorch container that stores multiple neural network modules (layers).
        #the below is like self.residial_connections = [
#     Residual_connection(dropout),   # index 0
#     Residual_connection(dropout)    # index 1 ]

        self.residual_connections = nn.ModuleList([Residual_connection(dropout) for _ in range(2)])
        
    def forward(self, x , src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask)) # lambda function takes x and immediately returns what is after :
        x = self.residual_connections[1](x, lambda x: self.feed_forward(x))
        return x
        

class Encoder(nn.Module):
    
    def __init__(self, layers = nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x , mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
        
# In encoder layer,
# nn.ModuleList was used inside a single layer
# to hold small internal modules., 
# In the Encoder example
# nn.ModuleList is used to build a stack of layers
# matching the Transformer architecture.
# Both usages are correct — just at different architectural levels.   


class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention_block : MultiheadAttention, cross_attention_block : MultiheadAttention, feed_forward_block: FeedForward, dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
        self.residual_connections = nn.ModuleList([Residual_connection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, trg_mask):
        # In DecoderBlock.forward
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, trg_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block(x))

        return x

class Decoder(nn.Module):
    
    def __init__(self, layers = nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x ,encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output , src_mask, trg_mask)
        return self.norm(x)
        
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model : int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self,x):
        #(batch, seq_len, d_model) --> (batch, seq_ln, vocab_size)
        
        return self.proj(x)     
        
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder:Decoder,src_embed: inputEmbeddings, trg_embed : inputEmbeddings,src_pos : PositionalEncoding, trg_pos : PositionalEncoding, projection_layer: ProjectionLayer ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed= trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
        
    def decode(self, encoder_output, trg, src_mask, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg,encoder_output, src_mask, trg_mask)
    
    def project(self, decoder_output):
        output = self.projection_layer(decoder_output)
        
        return output
        
def build_project(src_vocab_size:int, trg_vocab_size:int, src_seq_len:int, trg_seq_len:int, d_model:int = 512, N:int = 6, h:int = 8, dropout:float = 0.1, d_ff:int = 2058):
    #create embedding layers
    src_embed = inputEmbeddings(d_model, src_vocab_size)
    trg_embed = inputEmbeddings(d_model, trg_vocab_size)
    
    #create positionnal encoding layers
    
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model,trg_seq_len, dropout )
    encoder_states = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_states.append(encoder_block)
    
    #create decoder blocks
    
    decoder_states = []
    
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiheadAttention(d_model, h, dropout)
        decoder_feed_forward_block = FeedForward(d_model, d_ff,dropout )
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_states.append(decoder_block)
        
    #create encoder and decoder
    
    encoder = Encoder(nn.ModuleList(encoder_states))
    decoder = Decoder(nn.ModuleList(decoder_states))
    
    projection_layer = ProjectionLayer(d_model, trg_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos,projection_layer )
    
    #initialize the parameters
    
    for p in transformer.parameters():  #transformer.parameters() returns all trainable parameters in your Transformer model, including:
        if p.dim() > 1:    #p.dim() returns the number of dimensions of the tensor:
            nn.init.xavier_uniform_(p)   #Purpose: Scale weights so that the variance of activations is the same across layers.,Helps prevent vanishing/exploding gradients in deep networks.
    
    return transformer
    
def create_src_mask(src,pad_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return (src!=pad_idx).unsqueeze(1).unsqueeze(2).to(device)

def create_tgt_mask(tgt, pad_idx):
    """
    tgt: (batch, tgt_len)
    returns: (batch, 1, tgt_len, tgt_len)
    """
    device = tgt.device
    batch, tgt_len = tgt.shape

    # 1️ Padding mask: ignore pad tokens
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # shape: (B, 1, 1, tgt_len)

   
    
    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()  # shape: (tgt_len, tgt_len)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1,1,tgt_len,tgt_len)

    # 3️ Combine padding & causal masks
    return pad_mask & causal_mask  # shape: (B,1,tgt_len,tgt_len)


def get_model(src_vocab_size, tgt_vocab_size):
    model = build_project(src_vocab_size=src_vocab_size, trg_vocab_size=tgt_vocab_size, src_seq_len = 60, trg_seq_len = 60, d_model = 48, N=2, h=4, dropout = 0.1 )
    return model

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device {device}")
    model     = get_model(len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,           # ← reduced
        betas=(0.9, 0.98),
        eps=1e-9
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        n_batches  = 0
        
        for batch in dataloader:
            src    = batch["src"].to(device)
            tgt_in = batch["tgt_input"].to(device)
            tgt_out= batch["tgt_output"].to(device)
            
            src_mask = create_src_mask(src, pad_idx).to(device)
            tgt_mask = create_tgt_mask(tgt_in, pad_idx).to(device)
            
            enc_out = model.encode(src, src_mask)
            dec_out = model.decode(enc_out, tgt_in, src_mask, tgt_mask)
            output  = model.project(dec_out)
            
            output  = output.reshape(-1, output.size(-1))
            tgt_out = tgt_out.reshape(-1)
            
            loss = criterion(output, tgt_out)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← key fix
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches  += 1
        
        # Print AVERAGE epoch loss, not just last batch loss
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1} | Avg Loss = {avg_loss:.4f}")
    
    return model
   
model = train_model()    
     
    
def translate(sentence: str, model, src_vocab, tgt_vocab, 
              max_len=60, device="cpu") -> str:
    """Translate a sentence from English to French."""
    model.eval()
    
    # Reverse tgt_vocab to get id → word mapping
    idx2tgt = {v: k for k, v in tgt_vocab.items()}
    
    # Tokenize and encode source
    tokens     = tokenize(sentence)
    src_ids    = encode_src(tokens).unsqueeze(0).to(device)  # (1, seq_len)
    src_mask   = create_src_mask(src_ids, pad_idx)
    
    # Encode source
    with torch.no_grad():
        enc_out = model.encode(src_ids, src_mask)
    
    # Start decoder with <sos>
    tgt_ids = torch.tensor([[tgt_vocab["<sos>"]]], 
                            dtype=torch.long).to(device)
    
    output_tokens = []
    
    for _ in range(max_len):
        tgt_mask = create_tgt_mask(tgt_ids, pad_idx)
        
        with torch.no_grad():
            dec_out = model.decode(enc_out, tgt_ids, src_mask, tgt_mask)
            logits  = model.project(dec_out)         # (1, seq_len, vocab_size)
            next_id = logits[:, -1, :].argmax(dim=-1) # take last position
            
        print(
    "Predicted id:", next_id.item(),
    "Token:", idx2tgt.get(next_id.item())
)
        
        next_token = idx2tgt.get(next_id.item(), "<unk>")
        
        
        if next_token == "<eos>":
            break
        
        output_tokens.append(next_token)
        tgt_ids = torch.cat([tgt_ids, next_id.unsqueeze(0)], dim=1)
    
    return " ".join(output_tokens)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_sentence = input("Enter the sentence you want to translate    ")

print("\n Translated output is ",translate(input_sentence, model, src_vocab, tgt_vocab, device=device))


