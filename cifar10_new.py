import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.transforms import RandomErasing
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm # 引入进度条估计剩余时间
import os


# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 读取数据
batch_size = 128

def dataset(batch_size=16, valid_size=0.1, num_workers=0, datapath='datapath'):
    """"
    定义一个函数来读取cifar10
    batch_size: 批处理大小
    valid_size: 验证集占训练集的比重
    num_workers: 使用多少个子进程来加载数据
    datapath: 数据集存放路径
    """
    
    # 定义用于训练集的数据转换，运用数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 随机裁剪 32x32，添加4的边缘
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomRotation(15), # 随机旋转15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机改变颜色
        transforms.ToTensor(), # 将图像转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 标准化
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)) # 随机擦除
        ])
    
    # 定义用于测试集的数据转换
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 创建训练集和测试集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
   
    # 通过打乱索引划分训练集和验证集，保证不重不漏
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, valid_loader, test_loader 


# 读取数据
batch_size = 128
train_loader, valid_loader, test_loader = dataset(batch_size=batch_size, datapath='dataset')

# 加载ResNet18模型，并作小的修改
n_class = 10
model = models.resnet18()
# 把卷积改成3*3
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
# 将最后的全连接层改掉，添加一个 dropout 层
model.fc = nn.Sequential(
    nn.Dropout(0.3), # 在全连接层前添加 dropout 层，丢弃率为 0.3
    nn.Linear(512, n_class)
)
# 移除原版ResNet中conv1层后的最大池化层
model.maxpool = nn.Identity()
model = model.to(device) 
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)


# 训练部分
num_epochs = 20
lr = 0.001 # Adam对学习率不那么敏感
best_valid_loss = float('inf')
savepath = 'cifar10_new_checkpoint_2.pth' # 模型保存路径
start_epoch = 1 # 训练开始周期

# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
# 使用余弦退火改变学习率
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 断点续训
if os.path.exists(savepath):
    print(f"检测到保存点：{savepath}，正在加载...")
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1 # 恢复到上一个最佳损失的周期，而不是最新周期
    ave_train_loss = checkpoint['ave_train_loss']
    best_valid_loss = checkpoint['best_valid_loss']  # 恢复最佳验证集损失
    print(f"成功加载！将从第 {start_epoch} 个周期继续训练，上次训练集平均损失为 {ave_train_loss:.3f}，最佳验证集损失为 {best_valid_loss:.3f}")
else:
    print("未找到保存点，从头开始训练...")

    print(f"开始训练，总计 {num_epochs} 个周期...")    
    
for epoch in tqdm(range(1, num_epochs+1)):
    
    # 初始化损失值
    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    
    # 每100批次打印损失
    batch_count = 0
    batch_100_count = 0
    latest_loss = 0.0
    
    # 排版
    print("\n")
    
    # 训练模式
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 每次循环梯度清零
        output = model(data).to(device) # 前向传播
        loss = criterion(output, target) # 利用交叉熵计算当前批次损失
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        train_loss += loss.item() * data.size(0) # 将当前批次损失累加到当前周期损失
        latest_loss += loss.item() # 累加最近100个批次的损失
        batch_count += 1
        if batch_count % 100 == 0: # 每100个批次打印一次训练集平均损失
            batch_100_count += 1
            print(f" [周期 {epoch}, 批次 {batch_100_count*100}] 训练集最近100批次平均损失: {latest_loss / 100:.4f}")
            latest_loss = 0.0
            batch_count = 0
        
    # 一个epoch内，训练模式全部结束后，才进入验证模式
        
    # 验证模式
    model.eval()
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = model(data).to(device) # 前向传播
        loss = criterion(output, target) # 计算损失
        valid_loss += loss.item() * data.size(0) # 计算损失值
        _, pred = torch.max(output, 1) # 取最大值的索引
        total_sample += target.size(0) # 计算总样本数
        right_sample += (pred == target).sum().item() # 计算正确样本数
        
    # 计算平均损失和准确率
    ave_train_loss = train_loss / len(train_loader.sampler)
    ave_valid_loss = valid_loss / len(valid_loader.sampler)
    print(f' Epoch {epoch}, 训练集损失: {ave_train_loss:.4f}, 验证集损失: {ave_valid_loss:.4f}, 准确率: {100. * right_sample / total_sample:.2f}%')
    
    # 保存模型策略：判断损失函数是否下降
    if ave_valid_loss < best_valid_loss:
        best_valid_loss = ave_valid_loss
        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(), # 保存模型参数
                'optimizer_state_dict': optimizer.state_dict(), # 保存优化器参数
                'ave_train_loss': ave_train_loss, # 保存训练集损失
                'best_valid_loss': best_valid_loss # 保存验证集损失 
            }
        torch.save(checkpoint, savepath)
        print(f'※ 验证集损失下降，保存当前checkpoint，损失值：{best_valid_loss:.4f}')
    
    # 周期结束，学习率更新
    scheduler.step()
        
# 测试部分
model = models.resnet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = nn.Sequential(
    nn.Dropout(0.3), # 在全连接层前添加 dropout 层，丢弃率为 0.5
    nn.Linear(512, n_class)
)
model.maxpool = nn.Identity()# 移除最大池化层，保证测试集模型和训练集模型一致
model.load_state_dict(torch.load(savepath)['model_state_dict'])
model = model.to(device)

total_sample = 0
right_sample = 0
model.eval()
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data).to(device) # 前向传播
    _, pred = torch.max(output, 1) # 取最大值的索引
    total_sample += target.size(0) # 计算总样本数
    right_sample += (pred == target).sum().item() # 计算正确样本数
print(f'测试集准确率: {100. * right_sample / total_sample:.2f}%')