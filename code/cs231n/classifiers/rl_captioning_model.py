import numpy as np
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=512, hidden_dim=512, dtype=np.float32):
        super(PolicyNetwork, self).__init__()
        
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        
        vocab_size = len(word_to_idx)
        
        self.null = word_to_idx['<NULL>']
        self.start = word_to_idx.get('<START>', None)
        self.end = word_to_idx.get('<END>', None)
        
        self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)
        
        self.cnn2linear = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(wordvec_dim, hidden_dim, batch_first=True)
        self.linear2vocab = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, features, captions):
        input_captions = self.caption_embedding(captions)
        hidden_init = self.cnn2linear(features)
        hidden_init.unsqueeze(0)
        cell_init = torch.zeros_like(hidden_init)
        output, _ = self.lstm(input_captions, (hidden_init, cell_init))
        output = self.linear2vocab(output)
        return output
    
    
class ValueNetworkRNN(nn.Module):
    def __init__(self, input_dim=512, wordvec_dim=512, hidden_dim=512, dtype=np.float32):
        super(ValueNetworkRNN, self).__init__()
        
        self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)
        self.lstm = nn.LSTM(wordvec_dim, hidden_dim, batch_first=True)
    
    def forward(self, captions):
        input_captions = self.caption_embedding(captions)
        hidden_init = torch.zeros_like(512)
        cell_init = torch.zeros_like((1, 512))
        output, _ = self.lstm(input_captions, (hidden_init, cell_init))
        return output
    
class ValueNetwork(nn.Module):
    def __init__():
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 1)
    
    def forward(self, features, vnrnn):
        state = torch.cat((features, vrnn), dim=1)
        output = self.linear1(state)
        output = self.linear2(output)
        return output
    
class RewardNetworkRNN(nn.Module):
    def __init__(self, input_dim=512, wordvec_dim=512, hidden_dim=512, dtype=np.float32):
        super(RewardNetworkRNN, self).__init__()
        
        self.caption_embedding = nn.Embedding(vocab_size, wordvec_dim)
        self.gru = nn.GRU(wordvec_dim, hidden_dim, batch_first=True)
    
    def forward(self, captions):
        input_captions = self.caption_embedding(captions)
        hidden_init = torch.zeros_like(512)
        output, _ = self.gru(input_captions, hidden_init)
        return output
    
class RewardNetwork(nn.Module):
    def __init__():
        super(RewardNetwork, self).__init__():
        self.visual_embed = nn.Linear(512, 512)
        self.semantic_embed = nn.Linear(512, 512)
        
    def forward(self, features, captions):
        ve = self.visual_embed(features)
        se = self.semantic_embed(captions)
        return ve, se
        
