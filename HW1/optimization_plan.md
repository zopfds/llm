# COVID 患病率预测模型优化计划

## 当前模型分析

### 当前配置
- **模型结构**: 6层全连接网络 (93 → 64 → 64 → 64 → 64 → 64 → 1)
- **激活函数**: ReLU
- **损失函数**: MSE Loss
- **优化器**: SGD (lr=0.001, momentum=0.01)
- **批次大小**: 270
- **数据特征**: 93维，仅对第40列之后进行标准化
- **当前Loss**: ~0.9 (或更高)

### 问题诊断
1. **模型容量可能不足或过度**: 6层相同宽度可能不够灵活
2. **学习率可能不合适**: 0.001 对于 SGD 可能偏小
3. **缺少正则化**: 没有 Dropout、BatchNorm 等防止过拟合
4. **特征工程不足**: 只使用了部分特征的标准化
5. **没有学习率调度**: 固定学习率可能导致收敛慢
6. **优化器选择**: SGD 可能不如 Adam 等自适应优化器

---

## 详细优化计划

### 阶段 1: 模型架构优化

#### 1.1 改进网络结构
**原理**: 
- 当前模型使用相同宽度的层，表达能力有限
- 使用残差连接可以缓解梯度消失
- BatchNorm 可以加速训练并提高稳定性

**修改方案**:
```python
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        # 使用更合理的层宽度递减
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),  # 添加 BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),      # 添加 Dropout 防止过拟合
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
        self.criterion = nn.MSELoss(reduction='mean')
```

**预期效果**: Loss 降低 10-20%

---

#### 1.2 使用残差连接（可选）
**原理**: 
- 残差连接可以缓解深层网络的梯度消失问题
- 允许网络学习恒等映射，提高训练稳定性

**修改方案**:
```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        return nn.ReLU()(self.block(x) + x)  # 残差连接
```

---

### 阶段 2: 优化器与学习率优化

#### 2.1 更换为 Adam 优化器
**原理**:
- Adam 自适应调整每个参数的学习率
- 对超参数不敏感，收敛更快
- 结合了 Momentum 和 RMSprop 的优点

**修改方案**:
```python
config = {
    'optimizer': 'Adam',  # 从 SGD 改为 Adam
    'optim_hparas': {
        'lr': 0.001,      # 初始学习率
        'weight_decay': 1e-5  # L2 正则化
    }
}
```

**预期效果**: Loss 降低 15-25%

---

#### 2.2 添加学习率调度器
**原理**:
- 训练初期需要较大学习率快速收敛
- 训练后期需要较小学习率精细调整
- 学习率衰减可以避免在最优解附近震荡

**修改方案**:
```python
# 在 train 函数中添加
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50, verbose=True
)

# 在每个 epoch 后
scheduler.step(dev_mse)
```

**或使用余弦退火**:
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=n_epochs, eta_min=1e-6
)
```

**预期效果**: Loss 降低 5-10%

---

### 阶段 3: 数据预处理优化

#### 3.1 改进特征标准化
**原理**:
- 当前只对第40列之后标准化，前40列可能也需要处理
- 不同特征的量纲差异会影响模型学习

**修改方案**:
```python
# 在 COVID19Dataset 的 __init__ 中
# 对所有特征进行标准化（除了目标变量）
if mode != 'test':
    # 使用训练集的均值和标准差（避免数据泄露）
    self.data = (self.data - self.data.mean(dim=0, keepdim=True)) / \
                (self.data.std(dim=0, keepdim=True) + 1e-8)  # 防止除零
```

**预期效果**: Loss 降低 5-15%

---

#### 3.2 特征选择/特征工程
**原理**:
- 不是所有特征都重要
- 特征选择可以减少噪声，提高模型泛化能力

**修改方案**:
```python
# 使用相关性分析选择特征
import pandas as pd
train_df = pd.read_csv(tr_path)
correlation = train_df.corr()['target'].abs().sort_values(ascending=False)
important_features = correlation[correlation > 0.1].index.tolist()
```

**或使用 PCA 降维**:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)  # 保留前50个主成分
data_pca = pca.fit_transform(data)
```

**预期效果**: Loss 降低 3-8%

---

### 阶段 4: 损失函数优化

#### 4.1 使用 Huber Loss
**原理**:
- MSE 对异常值敏感
- Huber Loss 结合了 MSE 和 MAE 的优点
- 对异常值更鲁棒

**修改方案**:
```python
self.criterion = nn.HuberLoss(delta=1.0, reduction='mean')
```

**预期效果**: Loss 降低 3-5%

---

#### 4.2 使用 Smooth L1 Loss
**原理**:
- 在接近0时使用L2，远离0时使用L1
- 对异常值更鲁棒

**修改方案**:
```python
self.criterion = nn.SmoothL1Loss(reduction='mean')
```

---

### 阶段 5: 训练策略优化

#### 5.1 调整批次大小
**原理**:
- 批次大小影响梯度估计的稳定性
- 较小的批次可能提供更好的泛化
- 较大的批次可以加速训练

**修改方案**:
```python
config = {
    'batch_size': 64,  # 从 270 改为 64 或 128
}
```

**预期效果**: Loss 降低 2-5%

---

#### 5.2 添加梯度裁剪
**原理**:
- 防止梯度爆炸
- 提高训练稳定性

**修改方案**:
```python
# 在 train 函数中，backward() 之后
mse_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**预期效果**: 提高训练稳定性

---

#### 5.3 使用更长的训练时间
**原理**:
- 当前 early_stop=400 可能过早停止
- 深度学习模型需要充分训练

**修改方案**:
```python
config = {
    'n_epochs': 5000,      # 增加最大训练轮数
    'early_stop': 200,     # 调整早停策略
}
```

---

### 阶段 6: 集成学习（高级）

#### 6.1 模型集成
**原理**:
- 多个模型的平均预测通常比单个模型更准确
- 减少过拟合风险

**修改方案**:
```python
# 训练多个模型，然后平均预测
models = []
for seed in [42, 123, 456]:
    set_seed(seed)
    model = NeuralNet(input_dim).to(device)
    train(model, ...)
    models.append(model)

# 预测时
preds = []
for model in models:
    preds.append(model(x))
final_pred = torch.mean(torch.stack(preds), dim=0)
```

**预期效果**: Loss 降低 5-10%

---

## 实施优先级

### 高优先级（立即实施）
1. ✅ **更换为 Adam 优化器** - 效果最明显
2. ✅ **添加 BatchNorm 和 Dropout** - 防止过拟合
3. ✅ **改进网络结构** - 提高模型容量
4. ✅ **添加学习率调度** - 加速收敛

### 中优先级（第二波优化）
5. ✅ **改进特征标准化** - 数据预处理
6. ✅ **调整批次大小** - 训练策略
7. ✅ **使用 Huber Loss** - 损失函数优化

### 低优先级（进一步优化）
8. ✅ **特征选择** - 特征工程
9. ✅ **模型集成** - 高级技巧
10. ✅ **残差连接** - 架构优化

---

## 预期效果汇总

| 优化项 | 预期 Loss 降低 | 实施难度 |
|--------|---------------|----------|
| Adam 优化器 | 15-25% | 低 |
| BatchNorm + Dropout | 10-20% | 低 |
| 学习率调度 | 5-10% | 低 |
| 改进网络结构 | 10-20% | 中 |
| 特征标准化改进 | 5-15% | 中 |
| Huber Loss | 3-5% | 低 |
| 批次大小调整 | 2-5% | 低 |
| 模型集成 | 5-10% | 高 |

**总体预期**: 通过实施高优先级优化，Loss 可以从 0.9 降低到 **0.5-0.7** 左右

---

## 实施步骤

1. **第一步**: 实施 Adam 优化器 + 学习率调度
2. **第二步**: 添加 BatchNorm 和 Dropout
3. **第三步**: 改进网络结构（增加层数和宽度）
4. **第四步**: 改进数据预处理
5. **第五步**: 调整超参数（批次大小、训练轮数）
6. **第六步**: 尝试不同的损失函数
7. **第七步**: 如果还有提升空间，尝试模型集成

---

## 注意事项

1. **每次只改一个地方**，方便观察效果
2. **记录每次修改后的 Loss**，对比效果
3. **注意过拟合**：如果训练 Loss 远小于验证 Loss，需要增加正则化
4. **数据泄露**：确保测试集不参与任何训练过程
5. **随机种子**：保持固定以便复现结果
