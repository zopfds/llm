"""
优化后的 COVID 患病率预测模型
包含所有推荐的优化措施
"""

import torch
import torch.nn as nn
import numpy as np

# ==================== 优化后的模型架构 ====================

class OptimizedNeuralNet(nn.Module):
    """
    优化后的神经网络模型
    改进点：
    1. 更合理的网络结构（递减宽度）
    2. 添加 BatchNorm 加速训练
    3. 添加 Dropout 防止过拟合
    4. 使用 Huber Loss 提高鲁棒性
    """
    def __init__(self, input_dim, dropout_rate=0.3):
        super(OptimizedNeuralNet, self).__init__()
        
        self.net = nn.Sequential(
            # 第一层：扩大容量
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),  # BatchNorm 加速训练并提高稳定性
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout 防止过拟合
            
            # 第二层
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第三层
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.67),  # 逐渐减少 Dropout
            
            # 第四层
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.33),
            
            # 输出层（不使用 Dropout）
            nn.Linear(32, 1)
        )
        
        # 使用 Huber Loss 替代 MSE Loss（对异常值更鲁棒）
        self.criterion = nn.HuberLoss(delta=1.0, reduction='mean')
        # 或者使用 SmoothL1Loss
        # self.criterion = nn.SmoothL1Loss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


# ==================== 带残差连接的模型（可选） ====================

class ResidualBlock(nn.Module):
    """残差块，用于缓解梯度消失"""
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        return nn.ReLU()(self.block(x) + x)  # 残差连接


class ResidualNeuralNet(nn.Module):
    """带残差连接的神经网络"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        self.output_layer = nn.Linear(128, 1)
        self.criterion = nn.HuberLoss(delta=1.0, reduction='mean')
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x).squeeze(-1)
    
    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


# ==================== 优化后的训练函数 ====================

def train_optimized(tr_set, dv_set, model, config, device):
    """
    优化后的训练函数
    改进点：
    1. 使用 Adam 优化器
    2. 添加学习率调度器
    3. 添加梯度裁剪
    4. 改进早停策略
    """
    n_epochs = config['n_epochs']
    
    # 使用 Adam 优化器（替代 SGD）
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), 
        **config['optim_hparas']
    )
    
    # 学习率调度器：当验证损失不再下降时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # 监控验证损失的最小值
        factor=0.5,           # 学习率衰减因子
        patience=50,          # 等待50个epoch没有改善后降低学习率
        verbose=True,         # 打印学习率变化
        min_lr=1e-6          # 最小学习率
    )
    
    # 或者使用余弦退火调度器
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=n_epochs, eta_min=1e-6
    # )
    
    min_mse = float('inf')
    loss_record = {'train': [], 'dev': []}
    early_stop_count = 0
    best_epoch = 0
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            loss = model.cal_loss(pred, y)
            
            loss.backward()
            
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())
        
        avg_train_loss = np.mean(train_losses)
        loss_record['train'].append(avg_train_loss)
        
        # 验证阶段
        dev_mse = dev(dv_set, model, device)
        loss_record['dev'].append(dev_mse)
        
        # 学习率调度
        scheduler.step(dev_mse)
        
        # 保存最佳模型
        if dev_mse < min_mse:
            min_mse = dev_mse
            best_epoch = epoch + 1
            print(f'Saving model (epoch = {best_epoch:4d}, loss = {min_mse:.4f}, lr = {optimizer.param_groups[0]["lr"]:.6f})')
            torch.save(model.state_dict(), config['save_path'])
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        # 早停
        if early_stop_count > config['early_stop']:
            print(f'Early stopping at epoch {epoch + 1}!')
            print(f'Best model at epoch {best_epoch} with loss {min_mse:.4f}')
            break
        
        # 定期打印进度
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Dev Loss: {dev_mse:.4f}')
    
    print(f'Finished training after {epoch + 1} epochs')
    print(f'Best model at epoch {best_epoch} with loss {min_mse:.4f}')
    return min_mse, loss_record


# ==================== 改进的数据预处理 ====================

def improved_normalization(data, mode='train', train_stats=None):
    """
    改进的特征标准化
    对所有特征进行标准化，避免数据泄露
    """
    if mode == 'train':
        # 训练集：计算统计量并标准化
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True) + 1e-8  # 防止除零
        normalized = (data - mean) / std
        return normalized, mean, std
    else:
        # 验证/测试集：使用训练集的统计量
        mean, std = train_stats
        normalized = (data - mean) / std
        return normalized


# ==================== 优化后的配置 ====================

optimized_config = {
    'n_epochs': 5000,              # 增加最大训练轮数
    'batch_size': 64,              # 减小批次大小（从270改为64）
    'optimizer': 'Adam',            # 使用 Adam 优化器
    'optim_hparas': {
        'lr': 0.001,                # 初始学习率
        'weight_decay': 1e-5,      # L2 正则化
        'betas': (0.9, 0.999)      # Adam 的动量参数
    },
    'early_stop': 200,             # 早停轮数
    'save_path': 'models/optimized_model.pth'
}

# ==================== 使用示例 ====================

"""
使用优化后的模型：

# 1. 创建优化后的模型
model = OptimizedNeuralNet(input_dim=93, dropout_rate=0.3).to(device)

# 2. 使用优化后的训练函数
min_loss, loss_record = train_optimized(
    tr_set, dv_set, model, optimized_config, device
)

# 3. 绘制学习曲线
plot_learning_curve(loss_record, title='Optimized Model')

# 4. 加载最佳模型并预测
model.load_state_dict(torch.load(optimized_config['save_path']))
preds = test(tt_set, model, device)
"""
