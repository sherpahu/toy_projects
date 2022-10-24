# 概述
使用Self-Attention进行特征交叉
并且对连续特征也映射为Embedding向量（这个很奇怪）
![image.png](https://cdn.nlark.com/yuque/0/2022/png/705461/1666624110761-e0ca052f-5010-4097-9aa8-8d28e2d896fe.png#clientId=ucb5c0336-3f72-4&crop=0&crop=0&crop=1&crop=1&errorMessage=unknown%20error&from=paste&height=483&id=u22c679a0&margin=%5Bobject%20Object%5D&name=image.png&originHeight=725&originWidth=981&originalType=binary&ratio=1&rotation=0&showTitle=false&size=107752&status=error&style=none&taskId=u81fbd084-b56e-4b8a-a5ca-5719b9eadd5&title=&width=654)
# 背景
Transformer的self-attention在特征交叉上可能有用
# 方法
先将连续特征和类别特征表示为embedding向量，拼接起来
然后提出了Interacting Layer使用Multi-head Self-Attention进行特征的交叉
最后用于预估CTR

Self-Attention的细节
![](https://cdn.nlark.com/yuque/0/2022/png/705461/1666626827154-a2979303-f9b6-4dfa-8147-ec631867d184.png#clientId=u608e20b0-ec02-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=ua7b4f65d&margin=%5Bobject%20Object%5D&originHeight=478&originWidth=1187&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u306dd5b1-3b18-4584-a9c1-3bca9578cba&title=)

```python
def interacting_layer(x,head_num):
    dim = x.shape[-1]
    attention_output_dim = dim*x.shape[-2]
    Q = tf.keras.layers.Dense(units=head_num*dim)(x)
    K = tf.keras.layers.Dense(units=head_num*dim)(x)
    V = tf.keras.layers.Dense(units=head_num*dim)(x)
    Qs = tf.split(Q,head_num*[dim],-1)
    Ks = tf.split(K,head_num*[dim],-1)
    Vs = tf.split(V,head_num*[dim],-1)
    alphas = []
    for num in range(head_num):
        score = tf.nn.softmax(tf.matmul(Qs[num],Ks[num],transpose_b=True)/np.sqrt(dim))
        alpha = tf.matmul(score,Vs[num])
        alpha = tf.keras.layers.Flatten()(alpha)
        alphas.append(alpha)
    attention_output = tf.keras.layers.concatenate(alphas)
    attention_output = tf.keras.layers.Dense(units=attention_output_dim)(attention_output)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=x.shape[-1])(x)
    interact_layer_output = tf.keras.layers.Activation('relu')(tf.keras.layers.add([attention_output,x]))
    return interact_layer_output
```
```python
class InteractingLayer(nn.Module):
    def __init__(self, n_head, emb_size, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        
        self.w_qs = nn.Linear(emb_size, n_head * emb_size)
        self.w_ks = nn.Linear(emb_size, n_head * emb_size)
        self.w_vs = nn.Linear(emb_size, n_head * emb_size)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (emb_size + emb_size)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (emb_size + emb_size)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (emb_size + emb_size)))
        
        self.fc_attn = nn.Linear(n_head * emb_size, emb_size)
        nn.init.xavier_normal_(self.fc_attn.weight)

        self.fc_res = nn.Linear(n_head * emb_size, emb_size)
        nn.init.xavier_normal_(self.fc_res.weight)

    def forward(self, x, mask=None):
        residual = x
        q = rearrange(self.w_qs(x), 'b l (head k) -> head b l k', head=self.n_head)
        k = rearrange(self.w_ks(x), 'b t (head k) -> head b t k', head=self.n_head)
        v = rearrange(self.w_vs(x), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask[None], -np.inf)
        attn = torch.softmax(attn, dim=3)
        output = torch.einsum('hblt,hbtv->hblv', [attn, v])
        output = rearrange(output, 'head b l v -> b l (head v)')
        output = self.fc_attn(output)
        residual = self.fc_res(residual)
        output = F.relu(output+residual)
        return output
```
# 结论
成功应用了Self-Attention
