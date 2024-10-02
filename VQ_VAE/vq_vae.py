import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self._embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self._embedding.weight.data.normal_()
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = torch.Tensor(num_embeddings, self.embedding_dim)
        self._ema_w.data.normal_()
        self.decay = decay
        self.epsilon = epsilon
        
    def forward(self, inputs):
        # Change input shape from (B, C, H, W) to (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        # Flatten the inputs 
        flat_input = inputs.view(-1, self.embedding_dim)
        # Compute the distances between inputs and embeddings
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # Get the closest embeddings (quantization)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        
        # If training, update EMA
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self.epsilon) / 
                                      (n + self.num_embeddings * self.epsilon) * n)

            # Update weights
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = self._ema_w * self.decay + (1 - self.decay) * dw
            self._embedding.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)
            
            # Compute loss and apply straight-through estimator
            e_latent_loss = F.mse_loss(quantized.detach(), inputs)
            loss = self.commitment_cost * e_latent_loss
            
            # Straight-Through Estimator
            quantized = inputs + (quantized - inputs).detach()
            
            # Perplexity calculation
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            # Convert quantized back to (B, C, H, W)
            return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        
        return quantized.permute(0, 3, 1, 2).contiguous(), encodings


# Use VQ-VAE as  speech tokens to train LLM Model's 
num_embeddings = 256 
embedding_dim = 64    
commitment_cost = 0.25
decay = 0.99          
vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
batch_size = 8
channels = 1  
height = 128 
width = 128   
input_data = torch.randn(batch_size, channels, height, width)
loss, quantized_tokens, perplexity, encodings = vq_vae(input_data)
print("Quantized Tokens Shape:", quantized_tokens.shape)  
tokens_for_model = quantized_tokens.view(batch_size, -1)  
print("Tokens for Model Shape:", tokens_for_model.shape)

