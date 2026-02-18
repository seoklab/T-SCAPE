import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from src.utils_el import Tokenizer
from src.constants import PAD
from src.layers import PositionFeedForward, PositionFeedForward2d, DoubleEmbedding
from src.utils_el import one_hot_encode

class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class MaskedConv2d(nn.Conv2d):
    """ A masked 2-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv2D, except that the padding is set automatically.

         Shape:
            Input: (N, L, L, in_channels)
            input_mask: (N, L, L, 1), optional
            Output: (N, L, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()


class MaskedCausalConv1d(nn.Module):
    """Masked Causal 1D convolution based on https://github.com/Popgun-Labs/PopGen/. 
         
         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, groups=1, init=None):
        """
        Causal 1d convolutions with caching mechanism for O(L) generation,
        as described in the ByteNet paper (Kalchbrenner et al, 2016) and "Fast Wavenet" (Paine, 2016)
        Usage:
            At train time, API is same as regular convolution. `conv = CausalConv1d(...)`
            At inference time, set `conv.sequential = True` to enable activation caching, and feed
            sequence through step by step. Recurrent state is managed internally.
        References:
            - Neural Machine Translation in Linear Time: https://arxiv.org/abs/1610.10099
            - Fast Wavenet: https://arxiv.org/abs/1611.09482
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param init: optional initialisation function for nn.Conv1d module (e.g xavier)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups

        # if `true` enables fast generation
        self.sequential = False

        # compute required amount of padding to preserve the length
        self.zeros = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)

        # use supplied initialization function
        if init:
            init(self.conv)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        if input_mask is not None:
            x = x * input_mask
        # training mode
        x = torch.transpose(x, 1, 2)
        if not self.sequential:
            # no padding for kw=1
            if self.kernel_size == 1:
                return self.conv(x).transpose(1, 2)

            # left-pad + conv.
            out = self._pad(x)
            return self._unpad(self.conv(out)).transpose(1, 2)

        # sampling mode
        else:
            # note: x refers to a single timestep (batch, features, 1)
            if not hasattr(self, 'recurrent_state'):
                batch_size = x.size(0)
                self._init_recurrent_state(batch_size)

            return self._generate(x).transpose(1, 2)

    def _pad(self, x):
        return F.pad(x, [self.zeros, 0])

    def _unpad(self, x):
        return x

    def clear_cache(self):
        """
        Delete the recurrent state. Note: this should be called between runs, to prevent
        leftover state bleeding into future samples. Note that we delete state (instead of zeroing) to support
        changes in the inference time batch size.
        """
        if hasattr(self, 'recurrent_state'):
            del self.recurrent_state

    def _init_recurrent_state(self, batch):
        """
        Initialize the recurrent state for fast generation.
        :param batch: the batch size to generate
        """

        # extract weights and biases from nn.Conv1d module
        state = self.conv.state_dict()
        self.weight = state['weight']
        self.bias = state['bias']

        # initialize the recurrent states to zeros
        self.recurrent_state = torch.zeros(batch, self.in_channels, self.zeros, device=self.bias.device)

    def _generate(self, x_i):
        """
        Generate a single output activations, from the input activation
        and the cached recurrent state activations from previous steps.
        :param x_i: features of a single timestep (batch, in_channels, 1)
        :return: the next output value in the series (batch, out_channels, 1)
        """

        # if the kernel_size is greater than 1, use recurrent state.
        if self.kernel_size > 1:
            # extract the recurrent state and concat with input column
            recurrent_activations = self.recurrent_state[:, :, :self.zeros]
            f = torch.cat([recurrent_activations, x_i], 2)

            # update the cache for this layer
            self.recurrent_state = torch.cat(
                [self.recurrent_state[:, :, 1:], x_i], 2)
        else:
            f = x_i

        # perform convolution
        activations = F.conv1d(f, self.weight, self.bias,
                               dilation=self.dilation, groups=self.groups)

        return activations


class ByteNetBlock(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).
         
         Shape:
            Input: (N, L, d_in)
            input_mask: (N, L, 1), optional
            Output: (N, L, d_out)

    """

    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1, causal=False, activation='relu', rank=None):
        super().__init__()
        if causal:
            self.conv = MaskedCausalConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        else:
            self.conv = MaskedConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'gelu':
            act = nn.GELU
        layers1 = [
            nn.LayerNorm(d_in),
            act(),
            PositionFeedForward(d_in, d_h, rank=rank),
            nn.LayerNorm(d_h),
            act()
        ]
        layers2 = [
            nn.LayerNorm(d_h),
            act(),
            PositionFeedForward(d_h, d_out, rank=rank),
        ]
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        return x + self.sequence2(
            self.conv(self.sequence1(x), input_mask=input_mask)
        )


class ByteNet(nn.Module):

    """Stacked residual blocks from ByteNet paper defined by n_layers
         
         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)

    """

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.2, slim=True, activation='relu', down_embed=True):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()-
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        """
        super().__init__()
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Linear(n_tokens, d_embedding)
            else:
                self.embedder = DoubleEmbedding(n_tokens - n_frozen_embs, n_frozen_embs,
                                                d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()
        if down_embed:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert n_tokens == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x):
        e = self.embedder(x)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout, training=self.training)
        return e


class ByteNetLM(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.2, final_ln=True, slim=True, activation='relu',
                 tie_weights=True, down_embed=True):
        super().__init__()
        self.embedder = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
            self.decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

    def forward(self, x, input_mask=None):
        e = self.embedder(x, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)

class ConditionedByteNetDecoder(ByteNet):
    """ A conditioned, ByteNet decoder.
    Inputs:
        x (n, ell)
        c: (n, d_conditioning)

    """

    def __init__(self, n_tokens, d_embedding, conditional_d,  d_model, n_layers, kernel_size, r, mask_condition,
                 padding_idx=None, causal=False):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        """
        super().__init__(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                         padding_idx=padding_idx, causal=causal)
        self.mask_condition = mask_condition
        self.up_embedder = PositionFeedForward(d_embedding, d_model)
        self.ConditionedAttention = ConditionalAttention(d_model, d_model)
    def _embed(self, inputs):
        x, c = inputs
        e = self.embedder(x)
        e = self.up_embedder(e)
        e = self._convolve(e)
        # Concatenate the conditioning
        e = self.ConditionedAttention(e, c, self.mask_condition)
        return e
    
class ConditionedByteNetDecoder_MHC(ByteNet):
    """ A conditioned, ByteNet decoder.
    Inputs:
        x (n, ell)
        c: (n, d_conditioning)

    """

    def __init__(self, n_tokens, d_embedding, conditional_d,  d_model, n_layers, kernel_size, r, mask_condition,
                 padding_idx=None, causal=False):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        """
        super().__init__(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                         padding_idx=padding_idx, causal=causal)

        self.up_embedder = PositionFeedForward(d_embedding, d_model)
        self.ConditionedAttention = ConditionalAttentionMHC(d_model, d_model)
    def _embed(self, inputs):
        x, c = inputs
        e = self.embedder(x)
        e = self.up_embedder(e)
        e = self._convolve(e)
        # Concatenate the conditioning
        e = self.ConditionedAttention(e, c)
        return e

class ConditionalAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ConditionalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # Define the linear layers for Q, K, V
        self.query = nn.Linear(embedding_dim, hidden_dim)
        self.key = nn.Linear(1, hidden_dim)  # The scalar condition is used as the key
        self.value = nn.Linear(embedding_dim, hidden_dim)
        
    def forward(self, embeddings, condition, mask_condition = False):
        """
        embeddings: tensor of shape (batch_size, sequence_length, embedding_dim)
        condition: tensor of shape (batch_size, 1)
        """
        
        if mask_condition:
            #padding = torch.zeros([embeddings.shape[0],embeddings.shape[1], self.hidden_dim]).to(condition.device)
            return embeddings


        # Calculate Q, K, V
        Q = self.query(embeddings)
        K = self.key(condition)  # Unsqueeze to make it (batch_size, 1, hidden_dim)
        V = self.value(embeddings)
        
        # Compute attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # (batch_size, sequence_length, 1)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize over the sequence length
        
        # Compute the context vector
        context = torch.bmm(attention_weights.transpose(1, 2), V)  # (batch_size, 1, hidden_dim)
        
        # Concatenate the context vector to the original embeddings
        augmented_embeddings = torch.cat([embeddings, context.repeat(1, embeddings.shape[1], 1)], dim=-1)
        
        return context.repeat(1, embeddings.shape[1], 1)
    
class ConditionalAttentionMHC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ConditionalAttentionMHC, self).__init__()
        self.hidden_dim = hidden_dim

        # Adjust the dimensions for Q, K, V transformations
        # Assuming embedding_dim refers to the size of the embedding vectors (300 in your case)
        self.query = nn.Linear(embedding_dim, hidden_dim)
        self.key = nn.Linear(embedding_dim, hidden_dim)   # Adjust key processing to handle condition shape
        self.value = nn.Linear(embedding_dim, hidden_dim) # Process embeddings for value

    def forward(self, embeddings, condition):
        """
        embeddings: tensor of shape (batch_size, 20, embedding_dim)
        condition: tensor of shape (batch_size, 50, embedding_dim)
        """
        # Process Q, K, V
        Q = self.query(embeddings)  # (batch_size, 20, hidden_dim)
        K = self.key(condition)     # (batch_size, 50, hidden_dim)
        V = self.value(condition)   # Re-using condition as value but could be adjusted

        # Calculate attention scores: (batch_size, 20, 50)
        attention_scores = torch.bmm(Q, K.transpose(1, 2))

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights: (batch_size, 20, hidden_dim)
        context = torch.bmm(attention_weights, V)

        # Optional: Project context back to original embedding dimension if hidden_dim != embedding_dim
        if self.hidden_dim != embeddings.shape[-1]:
            projection = nn.Linear(self.hidden_dim, embeddings.shape[-1]).to(embeddings.device)
            context = projection(context)

        return context

class Sharedencoder(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.2, final_ln=False, slim=True, activation='relu',
                 tie_weights=True, down_embed=True):
        super().__init__()
        self.embedder_a = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        self.embedder_m1 = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        self.embedder_m2 = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

        self.d_embedding = d_embedding
        self.gate = nn.Linear(20,20,bias = False)
        tensor_dims = [20*d_model,50*d_model]
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.fc_layers = nn.ModuleList([nn.Linear(dim,d_model*20) for dim in tensor_dims])

    def forward(self,antigen,pseudo1, pseudo2, task):
        antigen = self.embedder_a(antigen)
        if task[0] in [3,4,5]:
            mhc_1 = self.embedder_m1(pseudo1)
            antigen = antigen.reshape(-1,20*self.d_model)
            mhc_1 = mhc_1.reshape(-1,50*self.d_model)
            projected_tensor = [fc(tensor) for fc, tensor in zip(self.fc_layers, [antigen,mhc_1])]
            projected_tensor = [tensor.view(-1,20,self.d_model) for tensor in projected_tensor]
            
            stacked_e = torch.stack(projected_tensor,dim=1)
            stacked_e = stacked_e.permute(0,1,3,2)
            gating_values = torch.sigmoid(self.gate(stacked_e))
            gated_e = gating_values * stacked_e
            e = torch.sum(gated_e, dim = 1)
            e = e.permute(0,2,1)
            return e
        elif task[0] in [6,7,8]:
            mhc_2 = self.embedder_m2(pseudo2)
            antigen = antigen.reshape(-1,20*self.d_model)
            mhc_2 = mhc_2.reshape(-1,50*self.d_model)
            projected_tensor = [fc(tensor) for fc, tensor in zip(self.fc_layers, [antigen,mhc_2])]
            projected_tensor = [tensor.view(-1,20,self.d_model) for tensor in projected_tensor]
            
            stacked_e = torch.stack(projected_tensor,dim=1)
            stacked_e = stacked_e.permute(0,1,3,2)
            gating_values = torch.sigmoid(self.gate(stacked_e))
            gated_e = gating_values * stacked_e
            e = torch.sum(gated_e, dim = 1)
            e = e.permute(0,2,1)
            return e
        else: 
            return antigen

class task1_encoder(nn.Module):
    def __init__(self, n_tokens, d_embedding, conditional_d, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.embedder = ConditionedByteNetDecoder(n_tokens, d_embedding, conditional_d, 
                                                  d_model, n_layers, kernel_size, r, mask_condition,
                                                  padding_idx=padding_idx, causal=causal)

    def forward(self, x, input_mask=None):
        seq, frac = x
        frac = frac.unsqueeze(-1)
        x = (seq, frac)
        e = self.embedder(x, input_mask=input_mask)
        return e

class task158_decoder(nn.Module):
    def __init__(self, d_model, n_tokens):
        super().__init__()
        self.decoder = PositionFeedForward(d_model*2, n_tokens)
    
    def forward(self, shared_e, task_e):
        e = torch.cat([shared_e, task_e], dim=-1)
        e = self.decoder(e)
        output = F.softmax(e,dim=2)
        return output
    
class task1(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task1_encoder = task1_encoder(n_tokens=n_tokens, d_embedding = d_embedding,conditional_d= 1, 
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal, mask_condition=mask_condition)
        self.task1_decoder = task158_decoder(d_model = d_model, n_tokens = n_tokens)

    def forward(self, p, frac):
        shared_e = self.shared_encoder(p)
        frac = frac.unsqueeze(-1)

        task1_e = self.task1_encoder((p,frac))
        output = self.task1_decoder(shared_e, task1_e)
        return shared_e, task1_e, output
    
class task3(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task3_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task3_decoder = task234679_decoder(d_model = d_model)
        
    def forward(self, p, mhc1, mhc2, tcr, task):
        shared_e = self.shared_encoder(p,mhc1, mhc2, task)
        task3_e = self.task3_encoder(p,mhc1)
        output = self.task3_decoder(shared_e, task3_e)
        return shared_e, task3_e, output
    
class task4(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task4_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task4_decoder = task234679_decoder(d_model = d_model)
        
    def forward(self, p, mhc1, mhc2, tcr, task):
        shared_e = self.shared_encoder(p,mhc1, mhc2, task)
        task4_e = self.task4_encoder(p,mhc1)
        output = self.task4_decoder(shared_e, task4_e)
        return shared_e, task4_e, output
    
class task6(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task6_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task6_decoder = task234679_decoder(d_model = d_model)
    
    def forward(self, p, mhc1, mhc2, tcr, task):
        shared_e = self.shared_encoder(p, mhc1, mhc2, task)
        task6_e = self.task6_encoder(p,mhc2)
        output = self.task6_decoder(shared_e, task6_e)
        return shared_e, task6_e, output
    
class task7(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task7_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task7_decoder = task234679_decoder(d_model = d_model)
    
    def forward(self, p, mhc1, mhc2, tcr, task):
        shared_e = self.shared_encoder(p, mhc1, mhc2, task)
        task7_e = self.task7_encoder(p,mhc2)
        output = self.task7_decoder(shared_e, task7_e)
        return shared_e, task7_e, output
    
class task9(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task9_encoder = task9_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task9_decoder = task234679_decoder(d_model = d_model)
    
    def forward(self, p, mhc1, mhc2, tcr, task):
        shared_e = self.shared_encoder(p, mhc1, mhc2, task)
        task9_e = self.task9_encoder(p,tcr)
        output = self.task9_decoder(shared_e, task9_e)
        return shared_e, task9_e, output

class task2_encoder(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.2, final_ln=True, slim=True, activation='relu',
                 tie_weights=True, down_embed=True):
        super().__init__()
        self.embedder = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
            self.decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

    def forward(self,x,input_mask=None):
        e = self.embedder(x, input_mask=input_mask)
        e = self.last_norm(e)
        return e

class task234679_decoder(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        # I want to use mlp as decoder (linear, relu, linear, relu, linear)
        self.decoder = nn.Sequential(
            nn.Linear(d_model*40, d_model*40),
            nn.ReLU(),
            nn.Linear(d_model*40, d_model*40),
            nn.ReLU(),
            nn.Linear(d_model*40,1))
        self.d_model = d_model
    
    def forward(self, shared_e, task_e):
        e = torch.cat([shared_e, task_e], dim=-1)
        e = e.reshape(-1,40*self.d_model)
        e = self.decoder(e)
        output = torch.sigmoid(e)
        return output

class task345678_encoder(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.2, final_ln=False, slim=True, activation='relu',
                 tie_weights=True, down_embed=True):
        super().__init__()
        self.embedder_a = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        self.embedder_m1 = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

        self.d_embedding = d_embedding
        self.gate = nn.Linear(20,20,bias = False)
        tensor_dims = [20*d_model,50*d_model]
        self.d_model = d_model
        self.fc_layers = nn.ModuleList([nn.Linear(dim,d_model*20) for dim in tensor_dims])

    def forward(self,antigen, mhc):
        antigen = self.embedder_a(antigen) # (batch, 20, d_model)
        mhc_1 = self.embedder_m1(mhc) # (batch, 50, d_model)
        antigen = antigen.reshape(-1,20*self.d_model)
        mhc_1 = mhc_1.reshape(-1,50*self.d_model)
        projected_tensor = [fc(tensor) for fc, tensor in zip(self.fc_layers, [antigen,mhc_1])]
        projected_tensor = [tensor.view(-1,20,self.d_model) for tensor in projected_tensor]

        stacked_e = torch.stack(projected_tensor, dim=0)
        stacked_e = stacked_e.permute(0,1,3,2)
        gating_values = torch.sigmoid(self.gate(stacked_e))
        gated_e = gating_values * stacked_e
        e = torch.sum(gated_e, dim=0)

        e = e.permute(0,2,1)
        return e
    
class task58_encoder(nn.Module):
    def __init__(self, n_tokens, d_embedding, conditional_d, d_model, n_layers, kernel_size, r,
                 dropout=0.2, down_embed=True, slim=True, activation='relu', rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.embedder = ConditionedByteNetDecoder_MHC(n_tokens = n_tokens, d_embedding= d_embedding,conditional_d= conditional_d,
                                                  d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r= r, mask_condition=mask_condition,
                                                  padding_idx=padding_idx, causal=causal)
        self.embedder_m1 = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        self.fc = nn.Linear(50*d_model, d_model)
        self.d_model = d_model

    def forward(self, seq, mhc, input_mask=None):
        e_m1 = self.embedder_m1(mhc)
        e_m1 = e_m1.reshape(-1,50*self.d_model)
        e_m1 = self.fc(e_m1)
        e_m1 = e_m1.view(-1,1,self.d_model)
        x = (seq, e_m1)
        e = self.embedder(x, input_mask=input_mask)
        return e


class task9_encoder(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.2, final_ln=False, slim=True, activation='relu',
                 tie_weights=True, down_embed=True):
        super().__init__()
        self.embedder_a = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        self.embedder_t2 = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

        self.d_embedding = d_embedding
        self.gate = nn.Linear(20,20,bias = False)
        tensor_dims = [20*d_model,25*d_model]
        self.d_model = d_model
        self.fc_layers = nn.ModuleList([nn.Linear(dim,d_model*20) for dim in tensor_dims])

    def forward(self,antigen, tcr_2):
        antigen = self.embedder_a(antigen) # (batch, 20, d_model)
        tcr_2 = self.embedder_t2(tcr_2) # (batch, 25, d_model)
        antigen = antigen.reshape(-1,20*self.d_model)
        tcr_2 = tcr_2.reshape(-1,25*self.d_model)
        projected_tensor = [fc(tensor) for fc, tensor in zip(self.fc_layers, [antigen,tcr_2])]
        projected_tensor = [tensor.view(-1,20,self.d_model) for tensor in projected_tensor]

        stacked_e = torch.stack(projected_tensor, dim=0)
        stacked_e = stacked_e.permute(0,1,3,2)
        gating_values = torch.sigmoid(self.gate(stacked_e))
        gated_e = gating_values * stacked_e
        e = torch.sum(gated_e, dim=0)

        e = e.permute(0,2,1)
        return e

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

class Merged_model(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, alphabet, device,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.device= device
        self.tokenizer = Tokenizer(alphabet)
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task1_encoder = task1_encoder(n_tokens=n_tokens, d_embedding = d_embedding,conditional_d= 1, 
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task1_decoder = task158_decoder(d_model = d_model,n_tokens=n_tokens)
        self.task2_encoder = task2_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task2_decoder = task234679_decoder(d_model = d_model)
        self.task3_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task3_decoder = task234679_decoder(d_model = d_model)
        
        self.task4_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        
        self.task4_decoder = task234679_decoder(d_model = d_model)
        self.task5_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task5_decoder = task158_decoder(d_model = d_model,n_tokens=n_tokens)
        
        self.task6_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        
        self.task6_decoder = task234679_decoder(d_model = d_model)
        
        self.task7_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task7_decoder = task234679_decoder(d_model = d_model)
        
        self.task8_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        
        self.task8_decoder = task158_decoder(d_model = d_model,n_tokens=n_tokens)
        
        self.task9_encoder = task9_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        
        self.task9_decoder = task234679_decoder(d_model = d_model)
        
        self.discriminator = Discriminator(input_size = d_model*20, hidden_size = d_model*5, output_size = 9)

    def forward(self, p, task, frac=[0.0] ,m1=PAD*50, m2=PAD*50, t2=PAD*25):
        if frac == [0.0]:
            frac = [0.0]*p.shape[0]
        if m1 == PAD*50:
            m1 = [PAD*50]*p.shape[0]
            m1 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in m1]
            m1 = torch.stack(m1)
            m1 = one_hot_encode(m1,29)
        if m2 == PAD*50:
            m2 = [PAD*50]*p.shape[0]
            m2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in m2]
            m2 = torch.stack(m2)
            m2 = one_hot_encode(m2,29)
        if t2 == PAD*25:
            t2 = [PAD*25]*p.shape[0]
            t2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in t2]
            t2 = torch.stack(t2)
            t2 = one_hot_encode(t2,29)
        frac = torch.FloatTensor(frac).unsqueeze(-1).to(self.device)
        m1 = m1.to(self.device)
        m2 = m2.to(self.device)
        t2 = t2.to(self.device)
        shared_e = self.shared_encoder(p, m1, m2, task)
        task1_e = self.task1_encoder((p,frac))
        task2_e = self.task2_encoder(p)
        task3_e = self.task3_encoder(p,m1)
        task4_e = self.task4_encoder(p,m1)
        task5_e = self.task5_encoder(p,m1)
        task6_e = self.task6_encoder(p,m2)
        task7_e = self.task7_encoder(p,m2)
        task8_e = self.task8_encoder(p,m2)
        task9_e = self.task9_encoder(p,t2)
        #task8_e = self.task8_encoder(p,m1)
        output_1 = self.task1_decoder(shared_e, task1_e)
        output_2 = self.task2_decoder(shared_e, task2_e)
        output_3 = self.task3_decoder(shared_e, task3_e)
        output_4 = self.task4_decoder(shared_e, task4_e)
        output_5 = self.task5_decoder(shared_e, task5_e)
        output_6 = self.task6_decoder(shared_e, task6_e)
        output_7 = self.task7_decoder(shared_e, task7_e)
        output_8 = self.task8_decoder(shared_e, task8_e)
        output_9 = self.task9_decoder(shared_e, task9_e)
        shared_e_flat = shared_e.reshape(-1,shared_e.shape[1]*shared_e.shape[2])
        reversed_shared_e = GradientReversalLayer.apply(shared_e_flat, 0.1)
        output = self.discriminator(reversed_shared_e)
        results = [(shared_e, task1_e, task2_e, task3_e, task4_e, task5_e, task6_e, task7_e, task8_e, task9_e),
                   (output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9),
                   output]
        return results
    
class Merged_model_mlm(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, alphabet, device,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.device= device
        self.tokenizer = Tokenizer(alphabet)
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task1_encoder = task1_encoder(n_tokens=n_tokens, d_embedding = d_embedding,conditional_d= 1, 
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task1_decoder = task158_decoder(d_model = d_model,n_tokens=n_tokens)
        self.task2_encoder = task2_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task2_decoder = task234679_decoder(d_model = d_model)
        self.task3_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task3_decoder = task234679_decoder(d_model = d_model)
        
        self.task4_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        
        self.task4_decoder = task158_decoder(d_model = d_model,n_tokens=n_tokens)
        self.task5_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task5_decoder = task234679_decoder(d_model = d_model)
        
        self.task6_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        
        self.task6_decoder = task158_decoder(d_model = d_model,n_tokens=n_tokens)
        
        self.task7_encoder = task9_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task7_decoder = task234679_decoder(d_model = d_model)
        
        self.discriminator = Discriminator(input_size = d_model*20, hidden_size = d_model*5, output_size = 9)

    def forward(self, p, task, frac=[0.0] ,m1=PAD*50, m2=PAD*50, t2=PAD*25):
        if frac == [0.0]:
            frac = [0.0]*p.shape[0]
        if m1 == PAD*50:
            m1 = [PAD*50]*p.shape[0]
            m1 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in m1]
            m1 = torch.stack(m1)
            m1 = one_hot_encode(m1,29)
        if m2 == PAD*50:
            m2 = [PAD*50]*p.shape[0]
            m2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in m2]
            m2 = torch.stack(m2)
            m2 = one_hot_encode(m2,29)
        if t2 == PAD*25:
            t2 = [PAD*25]*p.shape[0]
            t2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in t2]
            t2 = torch.stack(t2)
            t2 = one_hot_encode(t2,29)
        frac = torch.FloatTensor(frac).unsqueeze(-1).to(self.device)
        m1 = m1.to(self.device)
        m2 = m2.to(self.device)
        t2 = t2.to(self.device)
        shared_e = self.shared_encoder(p, m1, m2, task)
        task1_e = self.task1_encoder((p,frac))
        task2_e = self.task2_encoder(p)
        task3_e = self.task3_encoder(p,m1)
        task4_e = self.task4_encoder(p,m1)
        task5_e = self.task5_encoder(p,m2)
        task6_e = self.task6_encoder(p,m2)
        task7_e = self.task7_encoder(p,t2)
        output_1 = self.task1_decoder(shared_e, task1_e)
        output_2 = self.task2_decoder(shared_e, task2_e)
        output_3 = self.task3_decoder(shared_e, task3_e)
        output_4 = self.task4_decoder(shared_e, task4_e)
        output_5 = self.task5_decoder(shared_e, task5_e)
        output_6 = self.task6_decoder(shared_e, task6_e)
        output_7 = self.task7_decoder(shared_e, task7_e)
        shared_e_flat = shared_e.reshape(-1,shared_e.shape[1]*shared_e.shape[2])
        reversed_shared_e = GradientReversalLayer.apply(shared_e_flat, 0.1)
        output = self.discriminator(reversed_shared_e)
        results = [(shared_e, task1_e, task2_e, task3_e, task4_e, task5_e, task6_e, task7_e),
                   (output_1, output_2, output_3, output_4, output_5, output_6, output_7),
                   output]
        return results
    
class Merged_model_neg(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, alphabet, device,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.device= device
        self.tokenizer = Tokenizer(alphabet)
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task1_encoder = task1_encoder(n_tokens=n_tokens, d_embedding = d_embedding,conditional_d= 1, 
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task1_decoder = task158_decoder(d_model = d_model,n_tokens=n_tokens)
        self.task2_encoder = task2_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task2_decoder = task234679_decoder(d_model = d_model)
        self.task3_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task3_decoder = task234679_decoder(d_model = d_model)
        
        self.task4_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        
        self.task4_decoder = task234679_decoder(d_model = d_model)
        self.task5_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task5_decoder = task234679_decoder(d_model = d_model)
        
        self.task6_encoder = task345678_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        
        self.task6_decoder = task234679_decoder(d_model = d_model)
        
        self.task7_encoder = task9_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                      d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                      padding_idx=padding_idx, causal=causal)
        self.task7_decoder = task234679_decoder(d_model = d_model)
        
        self.discriminator = Discriminator(input_size = d_model*20, hidden_size = d_model*5, output_size = 9)

    def forward(self, p, task, frac=[0.0] ,m1=PAD*50, m2=PAD*50, t2=PAD*25):
        if frac == [0.0]:
            frac = [0.0]*p.shape[0]
        if m1 == PAD*50:
            m1 = [PAD*50]*p.shape[0]
            m1 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in m1]
            m1 = torch.stack(m1)
            m1 = one_hot_encode(m1,29)
        if m2 == PAD*50:
            m2 = [PAD*50]*p.shape[0]
            m2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in m2]
            m2 = torch.stack(m2)
            m2 = one_hot_encode(m2,29)
        if t2 == PAD*25:
            t2 = [PAD*25]*p.shape[0]
            t2 = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in t2]
            t2 = torch.stack(t2)
            t2 = one_hot_encode(t2,29)
        frac = torch.FloatTensor(frac).unsqueeze(-1).to(self.device)
        m1 = m1.to(self.device)
        m2 = m2.to(self.device)
        t2 = t2.to(self.device)
        shared_e = self.shared_encoder(p, m1, m2, task)
        task1_e = self.task1_encoder((p,frac))
        task2_e = self.task2_encoder(p)
        task3_e = self.task3_encoder(p,m1)
        task4_e = self.task4_encoder(p,m1)
        task5_e = self.task5_encoder(p,m1)
        task6_e = self.task6_encoder(p,m2)
        task7_e = self.task7_encoder(p,m2)
        output_1 = self.task1_decoder(shared_e, task1_e)
        output_2 = self.task2_decoder(shared_e, task2_e)
        output_3 = self.task3_decoder(shared_e, task3_e)
        output_4 = self.task4_decoder(shared_e, task4_e)
        output_5 = self.task5_decoder(shared_e, task5_e)
        output_6 = self.task6_decoder(shared_e, task6_e)
        output_7 = self.task7_decoder(shared_e, task7_e)
        shared_e_flat = shared_e.reshape(-1,shared_e.shape[1]*shared_e.shape[2])
        reversed_shared_e = GradientReversalLayer.apply(shared_e_flat, 0.1)
        output = self.discriminator(reversed_shared_e)
        results = [(shared_e, task1_e, task2_e, task3_e, task4_e, task5_e, task6_e, task7_e),
                   (output_1, output_2, output_3, output_4, output_5, output_6, output_7),
                   output]
        return results
'''  
class ablation_model(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model,d_conditioning, n_layers, kernel_size, r,task,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                                d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                                padding_idx = padding_idx, causal = causal)
        self.task = task
        self.discriminator = Discriminator(input_size = d_model*20, hidden_size = d_model*len(task), output_size = len(task))

        # task would be input as list
        # if task contain number 1
        if 1 in task:
            self.task1_encoder = task1_encoder(n_tokens=n_tokens, d_embedding = d_embedding,d_conditioning=d_conditioning,
                                                        d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                        padding_idx=padding_idx, causal=causal)
            self.task1_decoder = task1_decoder(d_model = d_model,n_tokens=n_tokens)
        # if task contain number 2
        if 2 in task:
            self.task2_encoder = task2_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                        d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                        padding_idx=padding_idx, causal=causal)
            self.task2_decoder = task2345_decoder(d_model = d_model)
        # if task contain number 3
        if 3 in task:
            self.task3_encoder = task34_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                        d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                        padding_idx=padding_idx, causal=causal)
            self.task3_decoder = task2345_decoder(d_model = d_model)
        # if task contain number 4
        if 4 in task:
            self.task4_encoder = task34_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                        d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                        padding_idx=padding_idx, causal=causal)
            self.task4_decoder = task2345_decoder(d_model = d_model)
        # if task contain number 5
        if 5 in task:
            self.task5_encoder = task5_encoder(n_tokens=n_tokens, d_embedding = d_embedding,
                                                        d_model = d_model, n_layers=n_layers, kernel_size=kernel_size, r=r,
                                                        padding_idx=padding_idx, causal=causal)
            self.task5_decoder = task2345_decoder(d_model = d_model)
            
    def forward(self, p, m1, m2, t1, t2,frac):
        shared_e = self.shared_encoder(p)
        shared_e_flat = shared_e.reshape(-1,shared_e.shape[1]*shared_e.shape[2])
        reversed_shared_e = GradientReversalLayer.apply(shared_e_flat, 0.1)
        output = []
        output.append(self.discriminator(reversed_shared_e))
        embeddings = []
        embeddings.append(shared_e)

        if 1 in self.task:
            task1_e = self.task1_encoder((p,frac))
            embeddings.append(task1_e)
            output.append(self.task1_decoder(shared_e, task1_e))
        else:
            embeddings.append(None)
            output.append(None)
        if 2 in self.task:
            task2_e = self.task2_encoder(p)
            embeddings.append(task2_e)
            output.append(self.task2_decoder(shared_e, task2_e))
        else:
            embeddings.append(None)
            output.append(None)
        if 3 in self.task:
            task3_e = self.task3_encoder(p,m1)
            embeddings.append(task3_e)
            output.append(self.task3_decoder(shared_e, task3_e))
        else:
            embeddings.append(None)
            output.append(None)
        if 4 in self.task:
            task4_e = self.task4_encoder(p,m2)
            embeddings.append(task4_e)
            output.append(self.task4_decoder(shared_e, task4_e))
        else:
            embeddings.append(None)
            output.append(None)
        if 5 in self.task:
            task5_e = self.task5_encoder(p,t2)
            embeddings.append(task5_e)
            output.append(self.task5_decoder(shared_e, task5_e))
        else:
            embeddings.append(None)
            output.append(None)
        
        
        return embeddings, output
'''  
class Finaltask1_perf(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task1_decoder1 = nn.Conv1d(d_model,d_model,kernel_size=1)
        self.task1_decoder2 = nn.Conv1d(d_model, d_model,kernel_size=1)
        self.task1_decoder3 = nn.Linear(d_model*5,1)
        self.d_model = d_model

    def forward(self, p, m1, m2,tcr, task):
        shared_e = self.shared_encoder(p, m1, m2, task=task)
        e = F.relu(self.task1_decoder1(shared_e.permute(0,2,1)))
        e = F.adaptive_max_pool2d(e,(self.d_model,10))
        e = F.relu(self.task1_decoder2(e))
        e = F.adaptive_max_pool2d(e,(self.d_model,5))
        e = self.task1_decoder3(e.reshape(-1, e.shape[1]*e.shape[2]))
        output = torch.sigmoid(e)
        return shared_e, output

class Finaltask1(nn.Module):
    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 padding_idx=None, causal=False, mask_condition = False):
        super().__init__()
        self.shared_encoder = Sharedencoder(n_tokens=n_tokens, d_embedding = d_embedding, 
                                            d_model = d_model, n_layers = n_layers, kernel_size = kernel_size, r = r,
                                            padding_idx = padding_idx, causal = causal)
        self.task1_decoder = nn.Linear(d_model*20,1)
        

    def forward(self, p):
        shared_e = self.shared_encoder(p)
        e = self.task1_decoder(shared_e.reshape(-1, shared_e.shape[1]*shared_e.shape[2]))
        output = torch.sigmoid(e)
        return shared_e, output
