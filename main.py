import spacy
import torch
import torch.nn as nn

from transformer import Transformer

if __name__ == "__main__":
    nlp = spacy.load("ru_core_news_sm")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Process whole documents
    with open("content/texts/poetic_data.txt", mode="r", encoding="utf-8") as file:
        text = file.read()
        text = text.replace("\n", " ")
        print("all data read")

    print("making doc")
    doc = nlp(text)

    print("making word2index")
    tokens = [token.text for token in doc[:]]
    bag_of_words = list(set(tokens))
    bag_of_words.sort()
    # bag_of_words.extend(["<pad>", "<sos>", "<eos>"])

    index2word = {index: word for index, word in enumerate(bag_of_words)}
    word2index = {word: index for index, word in enumerate(bag_of_words)}

    print("done.")

    bag_size = len(word2index.keys())
    token_dim = 1000
    enc_head_count = 10
    dec_head_count = 10
    batch_size = 5

    print("creating model...")
    model = Transformer(device, (1, 0, bag_size, token_dim, batch_size, batch_size), enc_head_count, dec_head_count).to(
        device)
    print("creating dataset...")
    dataset = [tokens[i:i + batch_size] for i in range(0, len(tokens), batch_size)]
    dataset = [(dataset[i], dataset[i + 1]) for i in range(0, len(dataset), 2)]

    model.train()
    torch.autograd.set_detect_anomaly(True)
    # print(list(model.parameters()))
    lr = 0.00001
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    for ep in range(1):
        for i in range(1000):  # len(text_parts) - 1:

            indexes = torch.LongTensor([word2index[word] for word in dataset[0][0]]).to(device)
            model_logits, model_idx = model(indexes)
            targets = torch.randn(model_logits.shape).to(device)
            # targets = torch.LongTensor([word2index[word] for word in dataset[1][1]])
            # targets = torch.eye(bag_size)[targets].to(device)
            #
            loss = loss_function(model_logits, targets)
            opt.zero_grad()
            loss.backward()
            #
            opt.step()
            print(loss.item())
            # for token in model_idx[:]:
            #     pass
            #     print(f"\"{index2word[token.item()]}\"", end=" ")

            params = opt.param_groups[0]["params"]
            print(len(list(filter(lambda x: x.grad is None, params))))
            # print(model.__repr__())
            # for p in opt.param_groups[0]['params']:
            #     print(p.grad)
            #     pass

        # print(nn.functional.cosine_similarity(sims[0], sims[1], dim=0))
