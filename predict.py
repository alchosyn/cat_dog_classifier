import main
import pretreat
import os
import torch
import pandas as pd

print("\n" + "=" * 30)
print("🚀 开始测试集推理以生成提交文件")
print("=" * 30)


# 1. 为测试集定义自定义 Dataset
class TestDataset(main.Dataset):
    """
    用于加载测试图像的自定义数据集。
    假设测试图像位于一个文件夹中，文件名为 '1.jpg', '2.jpg' 等。
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 获取图像文件名列表
        all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

        # 按其 ID 对文件进行数字排序 (例如, 1.jpg, 2.jpg, ... 10.jpg)
        try:
            # 假设文件ID是数字，如 "123.jpg" -> 123
            self.image_files = sorted(all_files, key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            print(f"警告: 无法按数字对测试文件排序。将按字母顺序排序。")
            self.image_files = sorted(all_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        # 加载图像
        try:
            image = main.Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"错误：加载图像 {img_path} 失败: {e}")
            return None, None  # 处理潜在的损坏图像

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 从文件名提取 ID (例如, "123.jpg" -> 123)
        img_id = int(os.path.splitext(img_name)[0])

        return image, img_id


# 2. 设置测试数据路径和 DataLoader
# 假设 'test' 文件夹与 'train' 和 'valid' 位于同一级别
test_dir = os.path.join(pretreat.data_dir, 'test')

# 对测试集使用验证集的变换
test_transform = pretreat.data_transforms['valid']

# 检查测试目录是否存在
if os.path.isdir(test_dir):
    test_dataset = TestDataset(data_dir=test_dir, transform=test_transform)
    # 使用 shuffle=False 来保持提交的顺序
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=pretreat.batch_size, shuffle=False)

    print(f"找到测试集: {len(test_dataset)} 张图像。")


    main.model_ft.eval()  # 将模型设置为评估模式

    all_ids = []
    all_preds = []

    with torch.no_grad():  # 禁用梯度计算
        for inputs, ids in test_dataloader:
            # 处理数据集中可能跳过的损坏图像
            if inputs is None or ids is None:
                continue

            inputs = inputs.to(pretreat.device)

            # 前向传播
            outputs = main.model_ft(inputs)

            # 获取预测 (0=cat, 1=dog，基于 ImageFolder 的自动标签)
            _, preds = torch.max(outputs, 1)

            # 存储结果
            all_ids.extend(ids.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 4. 创建并保存提交文件
    submission_df = pd.DataFrame({
        'ID': all_ids,
        'label': all_preds
    })

    # 按 ID 排序以确保顺序正确（尽管 DataLoader 应该已经保证了）
    submission_df = submission_df.sort_values(by='ID')

    submission_filename = "submission.csv"
    submission_df.to_csv(submission_filename, index=False)

    print(f"\n✅ 提交文件已创建: {submission_filename}")
    print("文件头部内容：")
    print(submission_df.head())

else:
    print(f"\n⚠️ 警告: 在 '{test_dir}' 未找到测试目录。")
    print("跳过测试集推理。")