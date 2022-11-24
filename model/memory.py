
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryNetwork(nn.Module):
    
    def __init__(self, 
		embedding_size, 
		feature_size,
		key_size, 
		memory_size, 
		hops, 
	):
    
        super().__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.key_size = key_size
        self.memory_size = memory_size
        self.hops = hops
        
        self.activation = nn.ReLU()
        self.query = nn.Parameter(
            torch.normal(mean=0.0, std=0.01, size=(1, self.embedding_size)), 
			requires_grad=True
        )
        self.a_embedding = nn.Parameter(
            torch.normal(mean=0.0, std=0.01, size=(self.embedding_size, self.feature_size)), 
			requires_grad=True
        )
        self.b_embedding = nn.Parameter(
            torch.normal(mean=0.0, std=0.01, size=(self.embedding_size, self.feature_size)), 
			requires_grad=True
        )
        self.c_embedding = nn.Parameter(
            torch.normal(mean=0.0, std=0.01, size=(self.embedding_size, self.feature_size)), 
			requires_grad=True
        )
        self.hidden = nn.Parameter(
            torch.normal(mean=0.0, std=0.01, size=(self.feature_size, self.feature_size)), 
			requires_grad=True
        )
        self.output = nn.Parameter(
            torch.normal(mean=0.0, std=0.01, size=(self.feature_size, self.key_size)), 
			requires_grad=True
        )

    def forward(self, stories):
        batch_size = stories.shape[0]

        u_0 = torch.mm(self.query, self.b_embedding)
        u_list = [u_0.repeat((batch_size, 1))]
        
        for _ in range(self.hops):
            
            m = torch.matmul(stories, self.a_embedding)                        
            u_temp = torch.unsqueeze(u_list[-1], -1)

            product = torch.bmm(m, u_temp).squeeze(-1)
            probs = F.softmax(product, 1)
            probs = torch.transpose(probs.unsqueeze(-1), 1, 2)

            c = torch.matmul(stories, self.c_embedding)
            c = torch.transpose(c, 1, 2)

            o = torch.sum(c * probs, dim=-1)
            h = torch.mm(o, self.hidden) + o
            u_k = self.activation(h)

            u_list.append(u_k)
                
        ret = torch.matmul(u_list[-1], self.output)    
        return ret
			