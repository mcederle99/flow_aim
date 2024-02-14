import torch
import torch.nn as nn
import torch.nn.functional as F

# LARGE model: input 512, nhead 8, feedforward 2048, num_layers 6
# MEDIUM model: input 256, nhead 4, feedforward 1024, num_layers 4
# SMALL model: input 128, nhead 2, feedforward 512, num_layers 2

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        self.encoder_emb = nn.Linear(state_dim, 256)
        self.decoder_emb = nn.Linear(state_dim+action_dim, 256)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=4, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=4)

        self.out_layer1 = nn.Linear(256, 128)
        self.out_layer2 = nn.Linear(128, 1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(self.device)        
    
    def forward(self, src):
        # src has shape (B, V, C)
        B, V, C = src.shape
        enc_src = self.encoder_emb(src)
        enc_tgt = self.encoder(enc_src)
        
        dec = torch.zeros((B, 1, C+1), dtype=torch.float32, device="cuda")
        actions = torch.tensor([], dtype=torch.float32, device="cuda")
        
        
        for i in range(V):
            dec_src = self.decoder_emb(dec)
        
            dec_tgt = self.decoder(dec_src, enc_tgt)
            dec_tgt = dec_tgt[:, -1, :]
            
            out = F.relu(self.out_layer1(dec_tgt))
            out = self.out_layer2(out)

            action = torch.tanh(out)
            
            tmp = torch.cat((src[:, i, :], action), dim=1).unsqueeze(dim=1)
            dec = torch.cat((dec, tmp), dim=1)

            actions = torch.cat((actions, action), dim=1)
        actions = actions.squeeze()
        if actions.shape == torch.Size():
            actions = actions.unsqueeze(dim=0)

        return actions

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.encoder_emb1 = nn.Linear(state_dim+action_dim, 256)

        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=4)

        self.fc1 = nn.Linear(256, 128)
        self.out1 = nn.Linear(128, 1)

        
        self.encoder_emb2 = nn.Linear(state_dim+action_dim, 256)

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, batch_first=True)
        self.encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=4)

        self.fc2 = nn.Linear(256, 128)
        self.out2 = nn.Linear(128, 1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(self.device)    
    
    def forward(self, state, action):
        src = torch.cat((state, action.unsqueeze(dim=-1)), dim=-1)
        # src has shape (B, V, C)
        B, V, C = src.shape

        enc_src1 = self.encoder_emb1(src)
        enc_tgt1 = self.encoder1(enc_src1)
                
        q1 = F.relu(self.fc1(enc_tgt1))
        q1 = self.out1(q1)
        q1 = q1.squeeze().sum(dim=1)
        
        ############################################################
        
        enc_src2 = self.encoder_emb2(src)
        enc_tgt2 = self.encoder2(enc_src2)
                
        q2 = F.relu(self.fc2(enc_tgt2))
        q2 = self.out2(q2)
        q2 = q2.squeeze().sum(dim=1)
        
        return q1, q2
    
    def Q1(self, state, action):
        src = torch.cat((state, action.unsqueeze(dim=-1)), dim=-1)
        # src has shape (B, V, C)
        B, V, C = src.shape

        enc_src1 = self.encoder_emb1(src)
        enc_tgt1 = self.encoder1(enc_src1)
                
        q1 = F.relu(self.fc1(enc_tgt1))
        q1 = self.out1(q1)
        q1 = q1.squeeze().sum(dim=1)
        
        return q1
