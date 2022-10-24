单头的Self-Attention
![](https://cdn.nlark.com/yuque/0/2022/png/705461/1666626943760-0a95f571-37ca-4088-8d10-61fc03392dab.png#clientId=uc1be2018-dd84-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=ucbd560ed&margin=%5Bobject%20Object%5D&originHeight=478&originWidth=1187&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u287dd5ca-f47b-449d-93e0-9fc6b73d819&title=)
```python
import torch
from einops import rearrange
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.1):
        super().__init__()
        self.w_qs=nn.Linear(in_dim,out_dim)
        self.w_ks=nn.Linear(in_dim,out_dim)
        self.w_vs=nn.Linear(in_dim,out_dim)
        
        self.fc=nn.Linear(out_dim,out_dim)
        self.dropout=nn.Dropout(dropout)
        self.layernorm=nn.LayerNorm(out_dim)
    def forward(self,x):
        dim=x.shape[-1]
        residual=x
        qs=self.w_qs(x)
        ks=self.w_ks(x)
        vs=self.w_vs(x)
        
        scaled_dot_prod=torch.einsum('b i w, b j w -> b i j', qs, ks)/(dim ** 0.5)
        
        score=torch.softmax(scaled_dot_prod,dim=-1)
        
        output=torch.einsum('b i w, b w j -> b i j', score, vs)
        output=self.dropout(self.fc(output))
        output=self.layernorm(output)
        return output

#输入dim=3
attention=SelfAttention(4,3)
#假设输入x为1x4x4矩阵（共4个token）
x = torch.rand(1,4,4)
attention(x)
```
```python
tensor([[[-0.8917,  1.3964, -0.5047],
         [-0.8896,  1.3968, -0.5072],
         [-0.8916,  1.3964, -0.5048],
         [-0.8906,  1.3966, -0.5060]]], grad_fn=<NativeLayerNormBackward0>)
```
