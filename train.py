import torch
import torch.nn as nn
from tqdm import tqdm

from TorchLinearNetwork import TorchLinearNetwork


class TorchTeacher:
    def __init__(self, data=None, education_speed: float = 0.001,
                 epochs: int = 200, device='cpu', error_function='cross_entropy'):
        self.__dataset = data
        self.__education_speed = education_speed
        self.__epochs = epochs
        self.__device = device
        self.__loss_functions = {
            'cross_entropy': nn.CrossEntropyLoss()
        }
        self.__loss_model = self.__loss_functions[error_function]

    def teach_model(self, model: TorchLinearNetwork):
        print(f"Длинна тренировочных данных: {len(self.__dataset['train']) + len(self.__dataset['val'])}")
        print(f"Длинна тестовых данных: {len(self.__dataset['test'])}")

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        lr_list = []
        best_loss = None
        lr = 0.001
        opt = torch.optim.Adam(model.parameters(), lr=lr)
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
            train_loop = tqdm(self.__dataset['train'], leave=False)
            for x, targets in train_loop:
                x = x.reshape(-1, 28 * 28).to(self.__device)
                x[torch.logical_and(x >= 0, x < 0.15)] = 0

                targets = targets.reshape(-1).to(torch.int32)
                targets = torch.eye(10)[targets].to(self.__device)

                pred = model(x)
                loss = self.__loss_model(pred, targets)

                opt.zero_grad()
                loss.backward()

                opt.step()

                running_train_loss.append(loss.item())
                mean_train_loss = sum(running_train_loss) / len(running_train_loss)

                true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

                train_loop.set_description(f"Epoch [{epoch + 1}/{self.__epochs}], train_loss={mean_train_loss:.4f}")

            running_train_acc = true_answer / (len(self.__dataset['train']) * 64)
            train_loss.append(mean_train_loss)
            train_acc.append(running_train_acc)

            # Проверка модели (валидация)
            model.eval()
            with torch.no_grad():
                running_val_loss = []
                true_answer = 0
                for x, targets in self.__dataset['val']:
                    x = x.reshape(-1, 28 * 28).to(self.__device)
                    targets = targets.reshape(-1).to(torch.int32)
                    targets = torch.eye(10)[targets].to(self.__device)

                    pred = model(x)
                    loss = self.__loss_model(pred, targets)

                    running_val_loss.append(loss.item())
                    mean_val_loss = sum(running_val_loss) / len(running_val_loss)

                    true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

                running_val_acc = true_answer / (len(self.__dataset['val']) * 64)
                val_loss.append(mean_val_loss)
                val_acc.append(running_val_acc)

            print(
                f"Epoch [{epoch + 1}/{self.__epochs}], train_loss={mean_train_loss:.4f}, train_acc={running_train_acc:.4f}, "
                f"val_loss={mean_val_loss:.4f}, val_acc={running_val_acc:.4f}")

            lr_scheduler.step(mean_val_loss)
            lr = lr_scheduler.last_lr[-1]
            lr_list.append(lr)

            if best_loss is None:
                best_loss = mean_val_loss

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss

                torch.save(model.state_dict(), f"model_state_dict_epoch_{epoch + 1}.pt")
                print(f"На эпохе - {epoch + 1}, сохранена модель со значением функции потерь на валидации - "
                      f"{mean_val_loss:.4f}, lr: {lr}", end='\n\n')

    def get_device(self):
        return self.__device
