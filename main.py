from tokenizer import BPETokenizer


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 加载分词器
    tokenizer = BPETokenizer()
    tokenizer.load('./bpe_tokenizer')

    # 测试分词和还原
    text = "且说鲁智深自离了五台山文殊院，取路投东京来，行了半月之上。"
    ids = tokenizer.encode(text)
    print("Encoded:", ids)
    print("Decoded:", tokenizer.decode(ids))

    print("\nVisualization:")
    print(tokenizer.print_visualization(text))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
