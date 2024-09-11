import torch
import torch.nn as nn
from tqdm import tqdm

import AbstractLinearNetwork
from data_load import TransformerTextLoader


class TorchTeacher:
    def __init__(self,
                 loader: TransformerTextLoader,
                 education_speed: float = 0.001,
                 epochs: int = 200,
                 device='cpu',
                 error_function='cross_entropy'):

        self.__loader = loader
        self.__education_speed = education_speed
        self.__epochs = epochs
        self.__device = device
        self.__loss_functions = {
            'cross_entropy': nn.CrossEntropyLoss()
        }
        self.__loss_model = self.__loss_functions[error_function]

    def teach_model(self, model: AbstractLinearNetwork):
        print(f"Длинна тренировочных данных: {len(self.__loader['train']) + len(self.__loader['val'])}")

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        lr_list = []
        best_loss = None
        opt = torch.optim.AdamW(model.parameters(), lr=self.__education_speed)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode='min',
            patience=5,
            factor=0.5
        )

        for epoch in range(self.__epochs):
            model.train()
            running_train_loss = []
            true_answer = 0
            train_loop = tqdm(self.__loader['train'], leave=False)

            for sample, target in train_loop:
                sample = self.__loader.convert_sample(sample, self.__device)
                target = self.__loader.convert_target(target, self.__device)
                predict, loss = model(sample, target=target, loss_function=self.__loss_model)
                predict = nn.functional.softmax(predict, dim=1)
                predict = torch.argmax(predict, dim=2)

                opt.zero_grad()
                loss.backward()
                #print(loss)
                count = 0
                for p in opt.param_groups[0]['params']:
                    if (p.grad is None):
                        count +=1
                print(count)
                exit(0)

                opt.step()

                running_train_loss.append(loss.item())
                mean_train_loss = sum(running_train_loss) / len(running_train_loss)

                true_answer += (predict == target.argmax(dim=2)).sum().item()

                #train_loop.set_description(f"Epoch [{epoch + 1}/{self.__epochs}], train_loss={mean_train_loss:.4f}")

            running_train_acc = true_answer / (len(self.__loader['train']) * self.__loader.get_batch_size())
            train_loss.append(mean_train_loss)
            train_acc.append(running_train_acc)
            print(train_acc[-1])


            # Проверка модели (валидация)
            # model.eval()
            # with torch.inference_mode():
            #     running_val_loss = []
            #     true_answer = 0
            #     for sample, target in self.__loader["val"]:
            #         sample = self.__loader.convert_sample(sample, self.__device)
            #         target = self.__loader.convert_target(target, self.__device)
            #         predict, loss = model(sample, target=target, loss_function=self.__loss_model)
            #
            #         predict = nn.functional.softmax(predict, dim=1)
            #         predict = torch.argmax(predict, dim=2)
            #
            #         running_val_loss.append(loss.item())
            #         mean_val_loss = sum(running_val_loss) / len(running_val_loss)
            #
            #         true_answer += (predict == target.argmax(dim=2)).sum().item()

                #running_val_acc = true_answer / (len(self.__loader['val']) * 4)
                #val_loss.append(mean_val_loss)
                #val_acc.append(running_val_acc)

           # print(
           #     f"Epoch [{epoch + 1}/{self.__epochs}], train_loss={mean_train_loss:.4f}, train_acc={running_train_acc:.4f}, "
           #     f"val_loss={mean_val_loss:.4f}, val_acc={running_val_acc:.4f}")

            # lr_scheduler.step(mean_val_loss)
            # #lr_list.append(self.__education_speed)
            #
            # if best_loss is None:
            #     best_loss = mean_val_loss
            #
            # if mean_val_loss < best_loss:
            #     best_loss = mean_val_loss
            #     torch.save(model.state_dict(), f"model_state_dict_epoch_{epoch + 1}.pt")
            #     print(f"На эпохе - {epoch + 1}, сохранена модель со значением функции потерь на валидации - "
            #           f"{mean_val_loss:.4f}, lr: {self.__education_speed}", end='\n\n')

    def get_device(self):
        return self.__device
