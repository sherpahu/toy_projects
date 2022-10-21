# 代码
我的：[https://github.com/sherpahu/toy_projects/tree/main/CoAtNet](https://github.com/sherpahu/toy_projects/tree/main/CoAtNet)
参考：
[https://github.com/chinhsuanwu/coatnet-pytorch/blob/master/coatnet.py](https://github.com/chinhsuanwu/coatnet-pytorch/blob/master/coatnet.py)
[https://mp.weixin.qq.com/s/LUA4zykI8IUX_s-MQvf-vw](https://mp.weixin.qq.com/s/LUA4zykI8IUX_s-MQvf-vw)
# CoAtNet
## 概述
CoAtNet结合卷积和transformer的self-attention，兼顾前者的泛化性与后者所具有的大模型容量，在ImageNet数据集上排名即为靠前。
## 背景
卷积和self-attention都可以看作per-dimension weighted sum of values in a pre-defined receptive
field。
卷积在小规模数据集上有很好的效果，卷积所具有的归纳偏置（平移不变性、局部感受野）使得泛化性很好。
transformer的自注意力机制则有一个大的感受野，“权重”可以根据输入自适应地调节，因而更容易捕捉不同空间位置之间复杂的关系交互。参数需要大量数据集的训练，而且缺乏归纳偏置，在小数据集上容易过拟合。
CoAtNet系统研究如何将ConvNet和transformer结合起来并进行科学的堆叠，集成二者的优势于一体。
## 方法
### MBConv
卷积操作是具有翻译等效性，可以在小数据集上面大大提升模型的泛化能力
![image.png](https://cdn.nlark.com/yuque/0/2022/png/705461/1666334488953-5db281da-b6ae-4469-ae58-8e6a667b51c1.png#clientId=u8b23b10e-e6a2-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=185&id=ub4314e44&margin=%5Bobject%20Object%5D&name=image.png&originHeight=278&originWidth=1615&originalType=binary&ratio=1&rotation=0&showTitle=false&size=58017&status=done&style=none&taskId=u93c57ce9-ad8c-40a8-b047-fefe95a5947&title=&width=1076.6666666666667)
深度卷积对于任意position pair，只要权重只与i-j有关，而与、的具体值无关，这被称为翻译等效性。
但是，卷积的感受野大小有限，限制了模型的容量和能力。

MBConv是在EfficientNet里面有使用，采用depthwise convolution和inverted residual bottlenecks

- 采用了Depthwise Convlution，因此相比于传统卷积，Depthwise Conv的参数能够大大减少；
- 采用了“倒瓶颈”的结构，也就是说在卷积过程中，特征经历了升维和降维两个步骤，这样做的目的应该是为了提高模型的学习能力。

![](https://cdn.nlark.com/yuque/0/2022/webp/705461/1666335922780-51911ced-b1dd-4c5c-ba80-a06445d8da7f.webp#clientId=u8b23b10e-e6a2-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=809&id=u70387e76&margin=%5Bobject%20Object%5D&originHeight=3218&originWidth=1440&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=uc6b3c95b-c944-4e5f-a927-0ebeb44e9ea&title=&width=362)
```python
class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample is False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)
```
### Relative Self-Attention

Self-Attention：

- 根据query和key计算相似度得到权重，可以使用点积、拼接等方法计算。
- 利用Softmax对计算得到的相似度权重进行归一化。
- 将归一化后的权重与value相乘，加权结果即为最终结果。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/705461/1666336692967-8eb77a70-ec87-4db1-b627-ab1917214db6.png#clientId=u8b23b10e-e6a2-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=329&id=u1243ae9a&margin=%5Bobject%20Object%5D&name=image.png&originHeight=360&originWidth=332&originalType=binary&ratio=1&rotation=0&showTitle=false&size=19158&status=done&style=none&taskId=u48b7e19a-b540-4fad-9d6c-f0468f7d4f9&title=&width=303.3333435058594)
self-attention可以使得感受野变为全局空间位置，可以学习更广泛更高维的concepts
![image.png](https://cdn.nlark.com/yuque/0/2022/png/705461/1666337168397-8aff6f5a-19df-4b8b-942e-503941eda9c2.png#clientId=u8b23b10e-e6a2-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=234&id=ua3dd0da9&margin=%5Bobject%20Object%5D&name=image.png&originHeight=351&originWidth=1644&originalType=binary&ratio=1&rotation=0&showTitle=false&size=77394&status=done&style=none&taskId=u218269cf-7683-44d6-a466-e8bf449b933&title=&width=1096)

CoAtNet使用input-independent version的relative attention，可以降低计算量，在inference的时候可以只算一次然后缓存使用。
```python
class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
```
```python
class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x
```
### 堆叠方式
将Self-attention和卷积的权重直接结合起来，放到softmax归一化之外或者里面
![image.png](https://cdn.nlark.com/yuque/0/2022/png/705461/1666337501577-405ae9c7-78d9-4f7a-bc7f-df6ab762ef2f.png#clientId=u8b23b10e-e6a2-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=115&id=ubde6ebe3&margin=%5Bobject%20Object%5D&name=image.png&originHeight=172&originWidth=1616&originalType=binary&ratio=1&rotation=0&showTitle=false&size=41515&status=done&style=none&taskId=u453cad51-b5f3-4069-a8d7-704a0679f29&title=&width=1077.3333333333333)
CoAtNet实际上用的是右边的 pre-normalization relative attention variant

- 先进行下采样，减少空间大小
- 在空间大小缩减下来之后，再采用全局的relative attention

堆叠方式为C-C-T-T ≈ C-T-T-T的样子效果最好
```python
class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)
```
## 结论
结合卷积和self-attention，兼顾泛化性和模型容量，并在ImageNet上取得了很好的结果。
自己在MNIST上尝试了一下，两个epoch达到了90%以上的正确率。
