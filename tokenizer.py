import jieba
from collections import Counter
import codecs

# 1. 读取文件并分词
def tokenize_file(input_file, output_file):
    with codecs.open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    tokenized_lines = []
    all_words = []

    for line in lines:
        # 使用结巴分词
        words = jieba.lcut(line.strip())
        # 将分词结果用空格连接
        tokenized_line = ' '.join(words)
        tokenized_lines.append(tokenized_line)
        all_words.extend(words)

    # 保存分词结果到新文件
    with codecs.open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tokenized_lines))

    return all_words

# 2. 词频统计并取前100个词
def get_top_words(words, top_n=1000):
    word_freq = Counter(words)
    top_words = word_freq.most_common(top_n)
    return top_words

# 3. 生成词表并保存token编码
def save_vocab(top_words, vocab_file):
    vocab = {word: [idx,  freq] for idx, (word, freq) in enumerate(top_words) if word!=" "}
    print(vocab)
    with codecs.open(vocab_file, 'w', encoding='utf-8') as f:
        for word, token in vocab.items():
            f.write(f"{word}\t{token[1]}\t{token[0]}\n")
    return vocab

# 主函数
def main():
    input_file = 'cn0.txt'          # 输入文件
    output_file = 'tokenized.txt'  # 分词后输出文件
    vocab_file = 'vocab.txt'       # 词表文件

    # 分词并保存
    print("正在分词并保存...")
    all_words = tokenize_file(input_file, output_file)

    # 词频统计
    print("正在统计词频...")
    top_words = get_top_words(all_words)

    # 生成词表并保存
    print("正在生成词表...")
    vocab = save_vocab(top_words, vocab_file)

    print("完成！")
    print("前10个高频词示例:")
    for word, freq in top_words[:]:
        print(f"{word}: {freq}")

    print(len(top_words))

# 执行
if __name__ == "__main__":
    main()
