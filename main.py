import numpy as np

# 1. 读取词表并构建反向映射和词频
def load_vocab(vocab_file):
    vocab,reverse_vocab,word_freq = {},{},{}
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            word, freq, token = line.strip().split('\t')
            token freq = int(token),float(freq)
            vocab[word] = token
            reverse_vocab[token] = word
            word_freq[token] = freq
    return vocab, reverse_vocab, word_freq

# 2. 自定义tokenizer（不填充）
def create_tokenizer(vocab, stop_words):
    def tokenize(text):
        words = text.strip().split()
        filtered_words = [word for word in words if word not in stop_words]
        tokens = [vocab.get(word, 0) for word in filtered_words]
        return np.array(tokens)
    return tokenize

# 3. 数据准备
def load_data(tokenized_file, vocab_file, seq_length, stop_words):
    vocab, reverse_vocab, word_freq = load_vocab(vocab_file)
    tokenizer = create_tokenizer(vocab, stop_words)
    with open(tokenized_file, 'r', encoding='utf-8') as f:
        train_texts = f.readlines()
    train_data = np.array([np.concatenate((tokenizer(text)[:seq_length],
                          np.zeros(seq_length - len(tokenizer(text)), dtype=int)))
                          if len(tokenizer(text)) < seq_length else tokenizer(text)[:seq_length]
                          for text in train_texts])
    labels = np.roll(train_data, -1, axis=1)
    freq_array = np.array([word_freq.get(i, 1.0) for i in range(vocab_size)])
    return train_data, labels, tokenizer, reverse_vocab, freq_array


# 4. 旋转位置编码
def rotary_pos_encoding(seq_len, dim, theta_base=10000):
    positions = np.arange(seq_len)
    dims = np.arange(dim // 2)
    theta = 1.0 / (theta_base ** (2 * dims / dim))
    angles = np.outer(positions, theta)
    sin,cos = np.sin(angles),np.cos(angles)
    return sin, cos

def apply_rotary(x, sin, cos):
    batch, seq, dim = x.shape
    # 如果输入序列超过预定义长度，扩展 sin 和 cos
    if seq > sin.shape[0]:
        sin_extended, cos_extended = rotary_pos_encoding(seq, dim)
        sin,cos = sin_extended, cos_extended
    else:
        sin,cos = sin[:seq], cos[:seq]
    x1, x2 = x[..., :dim//2], x[..., dim//2:]
    rot_x1 = x1 * cos[None, :, :] - x2 * sin[None, :, :]
    rot_x2 = x1 * sin[None, :, :] + x2 * cos[None, :, :]
    return np.concatenate([rot_x1, rot_x2], axis=-1)

# 5. 模型组件
class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.1

    def forward(self, x):
        return self.weights[x]

class AttentionHead:
    def __init__(self, embedding_dim, head_dim):
        self.q = np.random.randn(embedding_dim, head_dim) * 0.1
        self.k = np.random.randn(embedding_dim, head_dim) * 0.1
        self.v = np.random.randn(embedding_dim, head_dim) * 0.1
        self.head_dim = head_dim

    def forward(self, x, sin, cos):
        queries = x @ self.q
        keys = x @ self.k
        values = x @ self.v
        queries_rot = apply_rotary(queries, sin, cos)
        keys_rot = apply_rotary(keys, sin, cos)
        scores = queries_rot @ np.transpose(keys_rot, (0, 2, 1)) / (self.head_dim ** 0.5)
        attn = np.exp(scores) / (np.sum(np.exp(scores), axis=-1, keepdims=True) + 1e-6)
        return attn @ values

class MultiHeadAttention:
    def __init__(self, embedding_dim, heads):
        self.heads = [AttentionHead(embedding_dim, embedding_dim // heads) for _ in range(heads)]
        self.out = np.random.randn(embedding_dim, embedding_dim) * 0.1

    def forward(self, x, sin, cos):
        head_outputs = [head.forward(x, sin, cos) for head in self.heads]
        concat = np.concatenate(head_outputs, axis=-1)
        return concat @ self.out

class FeedForward:
    def __init__(self, embedding_dim):
        self.w1 = np.random.randn(embedding_dim, embedding_dim * 4) * 0.1
        self.w2 = np.random.randn(embedding_dim * 4, embedding_dim) * 0.1

    def forward(self, x):
        hidden = np.maximum(x @ self.w1, 0)
        return hidden @ self.w2

class TransformerLayer:
    def __init__(self, embedding_dim, heads):
        self.attn = MultiHeadAttention(embedding_dim, heads)
        self.ff = FeedForward(embedding_dim)

    def forward(self, x, sin, cos):
        attn_out = self.attn.forward(x, sin, cos)
        x = x + attn_out
        ff_out = self.ff.forward(x)
        return x + ff_out

class GPT:
    def __init__(self, vocab_size, embedding_dim, heads, layers, seq_length):
        self.embed = Embedding(vocab_size, embedding_dim)
        self.layers = [TransformerLayer(embedding_dim, heads) for _ in range(layers)]
        self.final_w = np.random.randn(embedding_dim, vocab_size) * 0.1
        head_dim = embedding_dim // heads
        # 增加最大位置编码长度以支持生成时的动态长度
        self.max_pos_length = max(seq_length, 16)  # 例如 64
        self.pos_sin, self.pos_cos = rotary_pos_encoding(self.max_pos_length, head_dim)
        self.max_seq_length = seq_length

    def forward(self, x):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        seq_len = x.shape[1]
        sin = self.pos_sin[:seq_len]
        cos = self.pos_cos[:seq_len]
        x = self.embed.forward(x)
        for layer in self.layers:
            x = layer.forward(x, sin, cos)
        logits = x @ self.final_w
        return logits

# 6. 训练代码
def train(model, data, labels, freq_array, epochs=100, lr=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(data)):
            x,y = data[i:i+1], labels[i:i+1]
            logits = model.forward(x)
            adjusted_logits = logits - np.log(freq_array + 1e-6)
            probs = np.exp(adjusted_logits) / np.sum(np.exp(adjusted_logits), axis=-1, keepdims=True)
            loss = -np.log(probs[0, range(seq_length), y[0]] + 1e-6).mean()
            total_loss += loss

            grad_logits = probs.copy()
            grad_logits[0, range(seq_length), y[0]] -= 1
            grad_final_w = model.embed.forward(x).transpose(0, 2, 1) @ grad_logits
            model.final_w -= lr * grad_final_w[0]

            grad_embed = grad_logits @ model.final_w.T
            embed_input = x[0]
            for t in range(seq_length):
                if embed_input[t] != 0:
                    model.embed.weights[embed_input[t]] -= lr * grad_embed[0, t]

        if epoch %10==0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(data)}")

# 7. 推理代码
def generate(model, start_text, tokenizer, reverse_vocab, freq_array, max_len=16, top_k=5):
    seq = tokenizer(start_text)
    print("初始序列:", [reverse_vocab[t] for t in seq if t in reverse_vocab])

    for step in range(max_len):
        logits = model.forward(seq)
        adjusted_logits = logits[0, -1] - np.log(freq_array + 1e-6)

        probs = np.exp(adjusted_logits) / np.sum(np.exp(adjusted_logits))

        current_seq = [reverse_vocab.get(t, "。") for t in seq]
        print(f"\n当前序列: {' '.join(current_seq)}")

        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]
        top_words = [reverse_vocab.get(idx, "。") for idx in top_indices]

        print(f"预测第 {step + 1} 个 token:")
        for word, prob in zip(top_words, top_probs):
            print(f"  {word}: {prob:.4f}")

        normalized_probs = top_probs / np.sum(top_probs)
        next_token = np.random.choice(top_indices, p=normalized_probs)
        print(next_token)
        seq = np.append(seq, next_token)

        if next_token==0:break ## 碰到句号就结束

    return seq

# 3. 数据准备
vocab_size = 100  ## 词表大小，目前设置为100，是经过加工之后的文本
seq_length = 16   ## 最大序列长度，目前设置为16
embedding_dim = 20  ## 词向量维度，目前设置为20
heads = 1          ## 注意力头数
layers = 5        ## transformer层数
stop_words = {"。"}  ## 碰到句号就结束

tokenized_file = 'cn.txt'
vocab_file = 'vocab.txt'
train_data, labels, tokenizer, reverse_vocab, freq_array = load_data(tokenized_file, vocab_file, seq_length, stop_words)

# 8. 执行
model = GPT(vocab_size, embedding_dim, heads, layers, seq_length)
print("开始训练...")
train(model, train_data, labels, freq_array, epochs=100,lr=0.01)

start_text = "你 今天 "
generated = generate(model, start_text, tokenizer, reverse_vocab, freq_array)
print("\n最终生成:", " ".join([reverse_vocab.get(t, "。") for t in generated]))
