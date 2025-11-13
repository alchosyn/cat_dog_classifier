# cifar_10_train.py
#
# 这是一个用于 Cifar-10 分类 (g) 和处理数据不平衡 (h) 的独立脚本。
#
# 完整功能包括:
# 1. (g) 训练一个 10 分类模型。
# 2. (g) 训练后，可视化 8 个样本的预测结果 (正确/错误)。
# 3. (h) 演示两种处理数据不平衡的方法。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights  # 使用新的 API
import numpy as np
import pandas as pd
import time
import copy
import os
import random
import matplotlib.pyplot as plt  # 导入可视化库

print(f"Torch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# --- (g) 和 (h) 任务的通用设置 ---

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Cifar-10 类别
class_names = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = 10
batch_size = 64
num_epochs = 1  # 为演示缩短 Epoch，你可以增加到 25 或更多

# (g) Cifar-10 图像尺寸调整
# ResNet 预训练输入为 224x224，但 Cifar-10 为 32x32
# 我们必须将图像放大，并使用 ImageNet 的均值和标准差
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪并放大到 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),  # 先放大到 256
        transforms.CenterCrop(224),  # 中心裁剪到 224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# --- 从你项目中复制的辅助函数 ---

def set_parameter_requires_grad(model, feature_extracting):
    """
    如果 feature_extracting = True，则冻结所有层
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    加载预训练模型并重置最后的全连接层。
    (从您的 main.py 复制并更新了 API)
    """
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        # 使用新的 'weights' API 来避免警告
        weights_param = None
        if use_pretrained:
            weights_param = ResNet18_Weights.DEFAULT

        model_ft = models.resnet18(weights=weights_param)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    """
    训练循环。
    (从您的 main.py 复制而来，并更新为记录所有历史)
    """
    since = time.time()

    # 初始化所有历史记录列表
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录所有四个指标
            # 深度复制模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # 使用 .cpu().item() 将 Tensor 转换为 float，以便绘图
            if phase == 'valid':
                val_acc_history.append(epoch_acc.cpu().item())
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc.cpu().item())
                train_losses.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)

    # 返回所有历史记录
    return model, train_losses, valid_losses, train_acc_history, val_acc_history


# --- (新增) 用于可视化的辅助函数 ---

def im_convert(tensor):
    """
    将 Tensor 图像反标准化并转换为可显示的 numpy 数组
    (从您的 main.py 复制而来)
    """
    image = tensor.to("cpu").clone().detach()  # 1. 复制 Tensor 到 CPU
    image = image.numpy().squeeze()  # 2. 转换为 NumPy 数组
    image = image.transpose(1, 2, 0)  # 3. 转换维度 (C,H,W) -> (H,W,C)

    # 4. 反标准化 (使用 ImageNet 均值和标准差)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    image = image.clip(0, 1)  # 5. 裁剪到 [0, 1] 范围
    return image


# --- (h) 用于模拟不平衡的辅助函数 ---

def create_unbalanced_dataset(full_dataset, minority_classes, reduction_factor=10):
    """
    从一个完整的数据集创建一个不平衡的数据集（Subset）。
    """
    print(f"\nCreating unbalanced dataset...")
    print(f"Minority classes: {[class_names[i] for i in minority_classes]} (Reduction Factor: {reduction_factor}x)")

    # 1. 获取所有目标的列表
    try:
        targets = full_dataset.targets
    except AttributeError:
        targets = [label for _, label in full_dataset]

    indices_to_keep = []
    class_counts = [0] * num_classes

    # 2. 迭代所有样本
    for i in range(len(full_dataset)):
        label = targets[i]
        if label in minority_classes:
            # 这是少数类，按概率保留
            if random.random() < (1.0 / reduction_factor):
                indices_to_keep.append(i)
                class_counts[label] += 1
        else:
            # 这是多数类，始终保留
            indices_to_keep.append(i)
            class_counts[label] += 1

    print("Unbalanced Class Counts:")
    for i in range(num_classes):
        print(f"  {class_names[i]:<10}: {class_counts[i]} samples")

    # 3. 创建一个 Subset
    unbalanced_subset = Subset(full_dataset, indices_to_keep)
    return unbalanced_subset, class_counts


# --- 主执行区 ---

def main():
    # =========================================================================
    # (g) 解决 Cifar-10 (平衡)
    # =========================================================================
    print("\n" + "=" * 30)
    print("🚀 (g) Task: Training on Cifar-10 (Balanced)")
    print("=" * 30)

    # 1. 加载平衡的 Cifar-10 数据集
    data_dir = './data'
    full_train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                          download=True, transform=data_transforms['train'])
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=data_transforms['valid'])

    # 2. 创建 Dataloaders (平衡)
    dataloaders_balanced = {
        'train': DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True),
        'valid': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    # 3. 初始化模型
    # 我们使用迁移学习 (feature_extract=True)
    model_g, _ = initialize_model("resnet18", num_classes, feature_extract=True, use_pretrained=True)
    model_g = model_g.to(device)

    # 4. 设置优化器和损失函数 (标准)
    # 只优化新添加的 fc 层的参数
    params_to_update_g = [param for param in model_g.parameters() if param.requires_grad]
    optimizer_g = optim.Adam(params_to_update_g, lr=0.001)

    criterion_g = nn.CrossEntropyLoss()  # 标准损失

    # 5. 训练
    # 'valid' 集就是 Cifar-10 的测试集，所以 "Best val Acc" 就是我们的测试集结果
    print("Training model for (g)...")
    model_g, train_losses_g, valid_losses_g, train_acc_g, valid_acc_g = train_model(
        model_g, dataloaders_balanced, criterion_g, optimizer_g, num_epochs=num_epochs
    )

    print("\n✅ (g) Task Complete. 'Best val Acc' is the result on the Cifar-10 testing set.")

    # =========================================================================
    # (g) 新增：(上一步已添加) 绘制训练历史折线图
    # =========================================================================
    print("\n" + "=" * 30)
    print("📊 正在为 (g) 任务生成训练历史折线图...")
    print("=" * 30)

    try:
        num_epochs_ran_g = len(train_losses_g)
        if num_epochs_ran_g > 0:
            epochs_range = range(1, num_epochs_ran_g + 1)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle('Task (g) Training History (Cifar-10)')

            # 绘制 损失 (Loss)
            ax1.plot(epochs_range, train_losses_g, 'b-o', label='Training Loss')
            ax1.plot(epochs_range, valid_losses_g, 'r-o', label='Validation Loss')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)

            # 绘制 准确率 (Accuracy)
            ax2.plot(epochs_range, train_acc_g, 'b-o', label='Training Accuracy')
            ax2.plot(epochs_range, valid_acc_g, 'r-o', label='Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局
            print("...请查看弹出的图表窗口。")
            plt.show()
            print("✅ (g) 任务折线图显示完毕。")
        else:
            print("⚠️ 警告: (g) 任务没有 epoch 数据可供绘图。")

    except Exception as e:
        print(f"⚠️ 警告: 绘制 (g) 任务图表时出错: {e}")

    # =========================================================================
    # (g) 新增：可视化分类结果
    # =========================================================================
    print("\n" + "=" * 30)
    print("📸 正在可视化 (g) 任务的分类结果...")
    print("=" * 30)
    print("绿色 = 正确, 红色 = 错误")
    print("格式: 预测结果 (真实标签)")
    print("...请查看弹出的窗口。关闭窗口后程序将继续执行 (h) 任务...")

    # 1. 从测试集（验证集）获取一个批次的数据
    dataiter = iter(dataloaders_balanced['valid'])
    images, labels = next(dataiter)

    # 2. 将模型设置为评估模式 (我们使用 (g) 任务训练好的 model_g)
    model_g.eval()

    # 3. 将图像传入模型获取预测结果
    if device.type == 'cuda':
        output = model_g(images.cuda())
    else:
        output = model_g(images)

    # 4. 获取预测的类别索引
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())

    # 5. 绘制图像和结果 (复制自您的 main.py)
    fig = plt.figure(figsize=(20, 10))  # 2行4列
    columns = 4
    rows = 2

    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        # 使用我们刚添加的 im_convert 函数
        ax.imshow(im_convert(images[idx]))

        # class_names 已在 cifar_10_train.py 顶部定义
        pred_label = class_names[preds[idx]]
        true_label = class_names[labels[idx].item()]

        # 判断对错并设置颜色
        is_correct = (pred_label == true_label)

        ax.set_title(
            f"{pred_label} ({true_label})",
            color=("green" if is_correct else "red")
        )

    # 6. 显示图像
    plt.show()  # 程序会在此暂停，直到您关闭窗口

    # =========================================================================
    # (g) 新增：将所有 10,000 张测试图片的结果保存为 CSV
    # =========================================================================
    print("\n" + "=" * 30)
    print("💾 正在为 (g) 任务生成完整的 CSV 预测文件...")
    print("=" * 30)
    print("...这可能需要一点时间，正在遍历所有 10,000 张测试图像...")

    # 1. 确保模型在评估模式
    model_g.eval()

    all_preds = []  # 存储所有预测
    all_true_labels = []  # 存储所有真实标签

    # 2. 禁用梯度，遍历 *整个* 测试集
    with torch.no_grad():  #

        # dataloaders_balanced['valid'] 包含整个 Cifar-10 测试集
        for inputs, labels in dataloaders_balanced['valid']:
            # 将数据移至设备 (GPU/CPU)
            inputs = inputs.to(device)

            # 运行模型
            outputs = model_g(inputs)

            # 获取预测结果
            _, preds_tensor = torch.max(outputs, 1)  #

            # 3. 将结果从 GPU/Tensor 转换回 CPU/numpy 并存储
            all_preds.extend(preds_tensor.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    print(f"已处理 {len(all_preds)} 张测试图像。")

    # 4. 将数字标签 (0-9) 转换为可读的类别名称
    # class_names 是在脚本顶部定义的 ('plane', 'car', ...)
    pred_names = [class_names[p] for p in all_preds]
    true_names = [class_names[t] for t in all_true_labels]

    # 5. 检查每个预测是否正确
    correct = (np.array(all_preds) == np.array(all_true_labels))

    # 6. 使用 Pandas 创建 DataFrame (类似 predict.py)
    results_df = pd.DataFrame({
        'ImageIndex': range(len(all_preds)),
        'PredictedLabel': pred_names,
        'TrueLabel': true_names,
        'IsCorrect': correct
    })

    # 7. 保存到 CSV 文件
    submission_filename = "cifar10_test_results.csv"
    results_df.to_csv(submission_filename, index=False)  #

    print(f"\n✅ 完整的预测结果已保存到: {submission_filename}")
    print("文件头部内容 (前5行)：")
    print(results_df.head())

    # --- [新功能] (g) 任务：绘制“类别准确率对比图” ---
    print("\n" + "=" * 30)
    print("📊 正在为 (g) 任务生成“类别准确率对比图”...")
    print("=" * 30)

    try:
        # 1. 初始化每个类别的正确数和总数
        # (我们使用 all_preds 和 all_true_labels, 它们是上一步刚生成的)
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        # 2. 遍历所有测试结果
        for i in range(len(all_true_labels)):
            true_label = all_true_labels[i]
            pred_label = all_preds[i]

            # 统计总数
            class_total[true_label] += 1

            # 统计正确数
            if true_label == pred_label:
                class_correct[true_label] += 1

        # 3. 计算每个类别的准确率
        per_class_accuracy = []
        print("Per-Class Accuracy (Task g):")
        for i in range(num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                per_class_accuracy.append(acc)
                print(f"  - {class_names[i]:<10}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                per_class_accuracy.append(0)
                print(f"  - {class_names[i]:<10}: N/A (0 samples)")

        # 4. 绘制条形图
        plt.figure(figsize=(15, 7))
        plt.bar(class_names, per_class_accuracy, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Accuracy (%)')
        plt.title('Per-Class Accuracy on Cifar-10 Test Set (Task g)')
        plt.ylim(0, 100)  # 准确率在 0% 到 100% 之间
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 在条形图上显示百分比
        for i, acc in enumerate(per_class_accuracy):
            plt.text(i, acc + 1, f'{acc:.1f}%', ha='center', color='black')

        print("\n...请查看弹出的“类别准确率对比图”窗口。")
        plt.show()
        print("✅ “类别准确率对比图”显示完毕。")

    except Exception as e:
        print(f"⚠️ 警告: 绘制“类别准确率对比图”时出错: {e}")
    # --- [新功能] 绘图代码结束 ---

    # =========================================================================
    # (h) 解决数据不平衡问题
    # =========================================================================
    print("\n" + "=" * 30)
    print("🚀 (h) Task: Handling Data Imbalance")
    print("=" * 30)

    # 1. 模拟一个不平衡的数据集
    # 我们让 'bird' (idx=2) 和 'ship' (idx=8) 成为少数类
    minority_classes = [2, 8]
    unbalanced_train_subset, class_counts = create_unbalanced_dataset(
        full_train_dataset,
        minorities=minority_classes,
        reduction_factor=10
    )

    # --- (h) 方法 1: 加权损失 (Weighted Loss) ---
    print("\n" + "-" * 20)
    print("Running (h) Approach 1: Weighted Loss")
    print("Justification: Penalizes errors on minority classes more heavily.")
    print("-" * 20)

    # 1.1 计算类别权重
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights.to(device)

    criterion_h1 = nn.CrossEntropyLoss(weight=weights)

    # 1.2 Dataloader (使用不平衡数据，但常规采样)
    dataloader_h1_train = DataLoader(unbalanced_train_subset, batch_size=batch_size, shuffle=True)
    dataloaders_h1 = {'train': dataloader_h1_train, 'valid': dataloaders_balanced['valid']}

    # 1.3 初始化新模型
    model_h1, _ = initialize_model("resnet18", num_classes, feature_extract=True, use_pretrained=True)
    model_h1 = model_h1.to(device)
    params_h1 = [p for p in model_h1.parameters() if p.requires_grad]
    optimizer_h1 = optim.Adam(params_h1, lr=0.001)

    # 1.4 训练
    print("Training model for (h) Approach 1...")
    model_h1, _, _, _, _ = train_model(model_h1, dataloaders_h1, criterion_h1, optimizer_h1, num_epochs=num_epochs)

    # --- (h) 方法 2: 加权随机采样 (Weighted Random Sampler) ---
    print("\n" + "-" * 20)
    print("Running (h) Approach 2: Weighted Random Sampler")
    print("Justification: Balances data at the batch level by oversampling minorities.")
    print("-" * 20)

    # 2.1 为 *每个样本* 计算权重
    subset_targets = [unbalanced_train_subset.dataset.targets[i] for i in unbalanced_train_subset.indices]
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[target] for target in subset_targets]

    # 2.2 创建 Sampler
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    # 2.3 Dataloader (使用 Sampler, **shuffle 必须为 False**)
    dataloader_h2_train = DataLoader(unbalanced_train_subset,
                                     batch_size=batch_size,
                                     sampler=sampler)  # shuffle=False (sampler 已处理随机性)

    dataloaders_h2 = {'train': dataloader_h2_train, 'valid': dataloaders_balanced['valid']}

    # 2.4 使用 *标准* 损失函数 (因为数据已经被采样平衡了)
    criterion_h2 = nn.CrossEntropyLoss()

    # 2.5 初始化新模型
    model_h2, _ = initialize_model("resnet18", num_classes, feature_extract=True, use_pretrained=True)
    model_h2 = model_h2.to(device)
    params_h2 = [p for p in model_h2.parameters() if p.requires_grad]
    optimizer_h2 = optim.Adam(params_h2, lr=0.001)

    # 2.6 训练
    print("Training model for (h) Approach 2...")
    model_h2, _, _, _, _ = train_model(model_h2, dataloaders_h2, criterion_h2, optimizer_h2, num_epochs=num_epochs)

    print("\n✅ (h) Task Complete. Both approaches have been demonstrated.")


if __name__ == '__main__':
    main()