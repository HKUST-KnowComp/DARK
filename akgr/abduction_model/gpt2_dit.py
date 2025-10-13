import torch.nn as nn
import torch

class DiffusionModel(nn.Module):
    """
    diffusion model
    """

    def __init__(
        self,
        model
    ):
        super().__init__()

        self.model = model
        self.embed_tokens = self.model.transformer.wte
        self.denoise_model = self.model.transformer # use inputs_embeds instead of input_ids in forward function
        for gpt2block in self.model.transformer.h:
            gpt2block.attn.bias.fill_(True)  # remove causal mask
        self.lm_head = self.model.lm_head
       
    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def get_embeds(self, input_ids):
        return self.embed_tokens(input_ids)
    
    def forward(self, input_ids,  attention_mask=None):
        """
        denoise the input
        """
        x_embed = self.get_embeds(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) 

        x = self.denoise_model(inputs_embeds = x_embed, attention_mask=attention_mask, return_dict = False)[0]

        logits = self.get_logits(x)

        return logits