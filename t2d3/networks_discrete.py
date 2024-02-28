import torch
import torch.nn as nn
import torch.nn.functional as F

# LARGE model: input 512, nhead 8, feedforward 2048, num_layers 4
# MEDIUM model: input 256, nhead 4, feedforward 1024, num_layers 2
# SMALL model: input 128, nhead 2, feedforward 512, num_layers 2

size = "S"

if size == "S":
    dim_model = 128
    num_head = 2
    ff = 512
    num_lay = 2
elif size == "M":
    dim_model = 256
    num_head = 4
    ff = 1024
    num_lay = 2
else:
    dim_model = 512
    num_head = 8
    ff = 2048
    num_lay = 4

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        self.encoder_emb = nn.Linear(state_dim, dim_model)
        self.decoder_emb = nn.Embedding(action_dim, dim_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_head, dim_feedforward=ff, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_lay)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_head, dim_feedforward=ff, dropout=0, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_lay)

        self.out_layer = nn.Linear(dim_model, action_dim)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(self.device)        
    
    def forward(self, src):
        # src has shape (B, V, C)
        B, V, C = src.shape
        enc_src = self.encoder_emb(src)
        enc_tgt = self.encoder(enc_src)
        
        dec = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        
        for i in range(V):
            dec_src = self.decoder_emb(dec)
            
            dec_tgt = self.decoder(dec_src, enc_tgt)
            dec_tgt = dec_tgt[:, -1, :]
            
            out = self.out_layer(dec_tgt)
            prob = F.softmax(out, dim=1)
            _, action = torch.max(prob, dim=1)
            action = action.unsqueeze(dim=-1)

            dec = torch.cat((dec, action), dim=1)
       
        actions = dec[:,1:].squeeze()

        if actions.shape == torch.Size():
            actions = actions.unsqueeze(dim=0)

        return actions

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.action_emb1 = nn.Embedding(action_dim, dim_model)
        self.encoder_emb1 = nn.Linear(state_dim, dim_model)

        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=dim_model*2, nhead=num_head, dim_feedforward=ff, dropout=0, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=num_lay)

        self.fc1 = nn.Linear(dim_model*2, dim_model)
        self.out1 = nn.Linear(dim_model, 1)

        self.action_emb2 = nn.Embedding(action_dim, dim_model) 
        self.encoder_emb2 = nn.Linear(state_dim, dim_model)

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=dim_model*2, nhead=num_head, dim_feedforward=ff, dropout=0, batch_first=True)
        self.encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=num_lay)

        self.fc2 = nn.Linear(dim_model*2, dim_model)
        self.out2 = nn.Linear(dim_model, 1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(self.device)    
    
    def forward(self, state, action):

        state_enc1 = self.encoder_emb1(state)
        action_enc1 = self.action_emb1(action)
        enc_src1 = torch.cat((state_enc1, action_enc1), dim=-1)

        enc_tgt1 = self.encoder1(enc_src1)
                
        q1 = F.relu(self.fc1(enc_tgt1))
        q1 = self.out1(q1)
        q1 = q1.squeeze().sum(dim=1)
        
        ############################################################
        
        state_enc2 = self.encoder_emb2(state)
        action_enc2 = self.action_emb2(action)

        enc_src2 = torch.cat((state_enc2, action_enc2), dim=-1)
        
        enc_tgt2 = self.encoder2(enc_src2)
                
        q2 = F.relu(self.fc2(enc_tgt2))
        q2 = self.out2(q2)
        q2 = q2.squeeze().sum(dim=1)
        
        return q1, q2
    
    def Q1(self, state, action):

        state_enc1 = self.encoder_emb1(state)
        action_enc1 = self.action_emb1(action)

        enc_src1 = torch.cat((state_enc1, action_enc1), dim=-1)

        enc_tgt1 = self.encoder1(enc_src1)
                
        q1 = F.relu(self.fc1(enc_tgt1))
        q1 = self.out1(q1)
        q1 = q1.squeeze().sum(dim=1)
        
        return q1
