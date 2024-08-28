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
    bag_of_words = list(set([token.text for token in doc[:]]))
    bag_of_words.sort()
    bag_of_words.extend(["<pad>", "<sos>", "<eos>"])

    index2word = {index: word for index, word in enumerate(bag_of_words)}
    word2index = {word: index for index, word in enumerate(bag_of_words)}

    print("done.")

    bag_size = len(word2index.keys())
    token_dim = 1000
    heads_count = 12
    print(bag_size)


    model = Transformer(device, {"self": (1, 1, bag_size, 25, 25),
                                 "encoder": (bag_size, token_dim, heads_count),
                                 "decoder": (1, token_dim, heads_count)}).to(device)

    with open("content/texts/poetic_data.txt", mode="r", encoding="utf-8") as file:
        text = file.read()
        text_parts = text.split("\n\n")
        model.train()
        lr = 0.000006
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()

        for i in range(len(text_parts) - 1): # len(text_parts) - 1:
            text_parts[i] = text_parts[i].replace("\n", " ")
            text_parts[i]: list[str] = [token.text for token in nlp(text_parts[i])]

            indexes = torch.LongTensor([word2index[word] for word in text_parts[i]]).to(device)

            model_logits, model_idx = model(indexes)

            targets = text_parts[i + 1].replace("\n", " ")
            targets = [token.text for token in nlp(targets)]
            targets.append("<eos>")
            targets.insert(0, "<sos>")
            targets = torch.LongTensor([word2index[word] for word in targets])
            targets = torch.eye(bag_size)[targets].to(device)

            batch_size = max(model_logits.size(dim=0), targets.size(dim=0))

            buff_tensor = torch.randn((batch_size - model_logits.size(dim=0), bag_size)).to(device)
            model_logits = torch.cat((model_logits, buff_tensor), dim=0)

            buff_tensor = torch.zeros(bag_size)
            buff_tensor[bag_size - 3] = 1.0
            buff_tensor = buff_tensor.expand((batch_size - targets.size(dim=0), bag_size)).to(device)
            targets = torch.cat((targets, buff_tensor), dim=0)

            # lol = [index2word[torch.argmax(tens).item()] for tens in targets[:]]
            # loss = loss_function(model_logits, targets)
            # opt.zero_grad()
            # loss.backward()
            #
            # opt.step()

            for token in model_idx[:]:
                pass
                print(f"\"{index2word[token.item()]}\"", end=" ")
            print()
