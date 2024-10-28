import math

import torch
import torch.nn as nn
from tqdm import tqdm

import AbstractLinearNetwork
from data_load import AbstractTransformerTextLoader


class LRScheduler:
    def __init__(self, max_learning_rate: float = 3e-4,
                 min_learning_rate: float = 1e-8,
                 warmup_steps: int = 20,
                 max_steps: int = 500):
        self.__max_learning_rate: float = max_learning_rate
        self.__min_learning_rate: float = min_learning_rate
        self.__warmup_steps: int = warmup_steps
        self.__max_steps: int = max_steps

    def get_lr(self, iteration: int) -> float:

        if iteration > self.__max_steps:
            return self.__min_learning_rate
        if iteration < self.__warmup_steps:
            return self.__max_learning_rate * (iteration + 1) / self.__warmup_steps

        decay_ratio = (iteration - self.__warmup_steps) / (self.__max_steps - self.__warmup_steps)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.__min_learning_rate + coefficient * (self.__max_learning_rate - self.__min_learning_rate)


class TorchTeacher:
    def __init__(self,
                 loader: AbstractTransformerTextLoader,
                 lr_scheduler: LRScheduler = LRScheduler(),
                 accumulation_steps: int = 2,
                 epochs: int = 200,
                 device='cpu',
                 error_function='cross_entropy'):

        self.__loader = loader
        self.__lr_scheduler = lr_scheduler
        self.__accumulation_steps = accumulation_steps
        self.__epochs: int = epochs
        self.__device: str = device
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
        opt = model.configure_optimizers(self.__device)
        iterations = 0

        for epoch in range(self.__epochs):
            model.train()
            running_train_loss = []
            true_answer = 0
            train_loop = tqdm(self.__loader['train'], leave=False)

            for batch_index, (sample, target) in enumerate(train_loop):
                sample: torch.LongTensor = sample.to(self.__device)
                target: torch.LongTensor = target.to(self.__device)
                with torch.autocast(device_type=self.__device, dtype=torch.float16):
                    predict, loss = model(sample, target=target, loss_function=self.__loss_model)
                    loss /= self.__accumulation_steps

                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                lr = self.__lr_scheduler.get_lr(iterations)
                for param_group in opt.param_groups:
                    param_group['lr'] = lr

                if batch_index % self.__accumulation_steps == 0:
                    opt.step()
                    opt.zero_grad()

                    running_train_loss.append(loss.item() * self.__accumulation_steps)
                    mean_train_loss = sum(running_train_loss) / len(running_train_loss)

                    predict = nn.functional.softmax(predict, dim=1)
                    predict = predict.argmax(dim=2)
                    true_answer += (predict == target).sum().item()

                    train_loop.set_description(
                        f"Epoch [{epoch + 1}/{self.__epochs}], train_loss={mean_train_loss:.4f}, norm={norm:.6f}, lr={lr:.6f}")
                    iterations += 1

            running_train_acc = true_answer / (len(self.__loader['train'])
                                               * self.__loader.get_batch_size()
                                               * predict.size(1))
            train_loss.append(mean_train_loss)
            train_acc.append(running_train_acc)

            # Проверка модели (валидация)
            model.eval()
            with torch.inference_mode():
                running_val_loss = []
                true_answer = 0
                for sample, target in self.__loader["val"]:
                    sample = sample.to(self.__device)
                    target = target.to(self.__device)
                    predict, loss = model(sample, target=target, loss_function=self.__loss_model)
                    predict = nn.functional.softmax(predict, dim=1)
                    predict = torch.argmax(predict, dim=2)
                    running_val_loss.append(loss.item())
                    mean_val_loss = sum(running_val_loss) / len(running_val_loss)
                    true_answer += (predict == target.argmax()).sum().item()

            running_val_acc = true_answer / (len(self.__loader['val']) * self.__loader.get_batch_size())
            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)

            print(
                f"Epoch [{epoch + 1}/{self.__epochs}], train_loss={mean_train_loss:.4f},"
                f" train_acc={running_train_acc:.4f},"
                f" "f"val_loss={mean_val_loss:.4f},"
                f" val_acc={running_val_acc:.4f}")

        # lr_scheduler.step(mean_val_loss)
        # #lr_list.append(self.__education_speed)

        if best_loss is None:
            best_loss = mean_val_loss
        #
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
        torch.save(model.state_dict(), f"model_state_dict_epoch_{epoch + 1}.pt")
        print(f"На эпохе - {epoch + 1}, сохранена модель со значением функции потерь на валидации - "
              f"{mean_val_loss:.4f}, lr: {self.__lr_scheduler.get_lr(epoch)}", end='\n\n')

    def get_device(self):
        return self.__device
