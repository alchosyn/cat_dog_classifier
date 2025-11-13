import os
import pretreat
import torch
import torch.nn as nn
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
def initialize_model(model_name, num_classes, feature_extract, use_pretrained = True):

    model_ft = pretreat.models.resnet18(pretrained = use_pretrained)
    pretreat.set_parameter_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)

    input_size = 64

    return model_ft, input_size


model_ft, input_size = initialize_model(pretreat.model_name, 2, pretreat.feature_extract, use_pretrained = True)


model_ft = model_ft.to(pretreat.device)
filename = "best.pt"
params_to_update = model_ft.parameters()
print("Params to learn:")
if pretreat.feature_extract:
    params_to_update =[]
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# optimizer
optimizer_ft = pretreat.optim.Adam(params_to_update, lr = 1e-2)
scheduler = pretreat.optim.lr_scheduler.StepLR(optimizer_ft, step_size = 10, gamma = 0.1)
criterion = nn.CrossEntropyLoss()

def train_model(model, dataloaders, criterion, optimizer, num_epochs =25, filename = 'best.pt'):
    since = time.time()
    best_acc = 0
    model.to(pretreat.device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []

    LRs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print("Epoch{}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train","valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(pretreat.device)
                labels = labels.to(pretreat.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()/len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print("Time elapsed {:.0f}m {:.0f}s".format(time_elapsed//60,time_elapsed % 60))
            print("{}Loss:{:.4f} Acc:{:.4f}".format(phase,epoch_loss,epoch_acc))

            if phase == "valid" and epoch_acc >best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    "state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == "valid":
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print("Optimizer learning rate:{:.7f}".format(optimizer.param_groups[0]["lr"]))
        LRs.append(optimizer.param_groups[0]["lr"])
        print()
        scheduler.step()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best val Acc:{}:4f".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs




for param in model_ft.parameters():
    param.requires_grad = True

optimizer = pretreat.optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = pretreat.optim.lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

criterion = nn.CrossEntropyLoss()

checkpoint = torch.load(filename)
best_acc = checkpoint["best_acc"]
model_ft.load_state_dict(checkpoint["state_dict"])

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, pretreat.dataloaders, criterion, optimizer_ft, num_epochs = 20)

print("\n" + "=" * 30)
print("正在保存训练历史记录")
print("=" * 30)

# 1. 确定实际运行的 Epoch 数量
num_epochs_ran = len(train_losses)


def to_float_list(history_list):
    float_list = []
    for item in history_list:
        if isinstance(item, torch.Tensor):
            float_list.append(item.cpu().item())
        else:
            float_list.append(float(item))
    return float_list


try:
    train_losses_float = to_float_list(train_losses)
    valid_losses_float = to_float_list(valid_losses)
    train_acc_float = to_float_list(train_acc_history)
    valid_acc_float = to_float_list(val_acc_history)

    # 创建 DataFrame
    history_df = pd.DataFrame({
        'epoch': range(1, num_epochs_ran + 1),
        'train_loss': train_losses_float,
        'valid_loss': valid_losses_float,
        'train_acc': train_acc_float,
        'valid_acc': valid_acc_float
    })

    history_csv_file = "training_history.csv"
    history_df.to_csv(history_csv_file, index=False)
    print(f"✅ 训练历史已保存到 (CSV): {history_csv_file}")
    print("历史记录预览：")
    print(history_df)

except Exception as e:
    print(f"⚠️ 警告: 保存为 CSV 时出错: {e}")

# --- 🔽 [新功能] 添加绘图代码 🔽 ---
try:
    print("\n" + "=" * 30)
    print("📊 正在生成训练历史折线图...")
    print("=" * 30)

    # 确保我们有 epoch 数据来绘图
    if num_epochs_ran > 0:
        # 创建图表 (2行, 1列)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Training History (Cat/Dog)')

        epochs_range = range(1, num_epochs_ran + 1)

        # 绘制 损失 (Loss)
        ax1.plot(epochs_range, train_losses_float, 'b-o', label='Training Loss')
        ax1.plot(epochs_range, valid_losses_float, 'r-o', label='Validation Loss')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # 绘制 准确率 (Accuracy)
        # (注意：你的代码使用 to_float_list 转换了 train_acc_float 和 valid_acc_float)
        ax2.plot(epochs_range, train_acc_float, 'b-o', label='Training Accuracy')
        ax2.plot(epochs_range, valid_acc_float, 'r-o', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为总标题留出空间
        print("...请查看弹出的图表窗口。")
        plt.show()
        print("✅ 折线图显示完毕。")
    else:
        print("⚠️ 警告: 没有 epoch 数据可供绘图。")

except Exception as e:
    print(f"⚠️ 警告: 绘制图表时出错: {e}")
# --- 🔼 [新功能] 绘图代码结束 🔼 ---


dataiter = iter(pretreat.dataloaders["valid"])
images, labels = next(dataiter)



def to_float_list(history_list):
    float_list = []
    for item in history_list:
        if isinstance(item, torch.Tensor):
            float_list.append(item.cpu().item())
        else:
            float_list.append(float(item))
    return float_list



try:
    train_losses_float = to_float_list(train_losses)
    valid_losses_float = to_float_list(valid_losses)
    train_acc_float = to_float_list(train_acc_history)
    valid_acc_float = to_float_list(val_acc_history)

    # 创建 DataFrame
    history_df = pd.DataFrame({
        'epoch': range(1, num_epochs_ran + 1),
        'train_loss': train_losses_float,
        'valid_loss': valid_losses_float,
        'train_acc': train_acc_float,
        'valid_acc': valid_acc_float
    })

    history_csv_file = "training_history.csv"
    history_df.to_csv(history_csv_file, index=False)
    print(f"✅ 训练历史已保存到 (CSV): {history_csv_file}")
    print("历史记录预览：")
    print(history_df)

except Exception as e:
    print(f"⚠️ 警告: 保存为 CSV 时出错: {e}")




dataiter = iter(pretreat.dataloaders["valid"])
images, labels = next(dataiter)

model_ft.eval()

if pretreat.train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)


_, preds_tensor = torch.max(output,1)
preds = np.squeeze(preds_tensor.numpy()) if not pretreat.train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())




def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229,0.224,0.225)) + np.array((0.485,0.456,0.406))
    image = image.clip(0,1)

    return image


fig = plt.figure(figsize=(20,20))
columns = 4
rows = 2

for idx in range(columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    ax.imshow(im_convert(images[idx]))
    ax.set_title(
        "{} ({})".format(pretreat.class_names[preds[idx]], pretreat.class_names[labels[idx].item()]),
        color=("green" if pretreat.class_names[preds[idx]] == pretreat.class_names[labels[idx].item()] else "red")
    )

plt.show()





