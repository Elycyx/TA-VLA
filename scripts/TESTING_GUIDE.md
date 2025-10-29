# 策略测试脚本使用指南

本目录包含三个测试脚本，用于测试运行中的 OpenPI 策略服务器。

## 快速开始

### 1. 启动策略服务器

首先，你需要启动策略服务器：

```bash
# 使用 FAVLA 配置和 checkpoint
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_lora_favla \
  --policy.dir=assets/pi0_lora_favla/cyx/forceumi1
```

服务器启动后会显示类似以下信息：
```
INFO:__main__:Creating server (host: your-hostname, ip: xxx.xxx.xxx.xxx)
```

### 2. 选择测试脚本

根据你的需求选择合适的测试脚本：

| 脚本 | 用途 | 适用场景 |
|------|------|----------|
| `test_policy_simple.py` | 快速测试 | 验证服务器是否正常工作 |
| `test_favla_policy.py` | 性能测试 | 测量推理速度和性能 |
| `test_policy_with_dataset.py` | 真实数据测试 | 使用实际数据集验证模型 |

---

## 脚本详细说明

### 1. test_policy_simple.py - 快速测试脚本

**用途**: 快速验证服务器是否正常工作，发送单个 dummy 观测并查看输出。

**特点**:
- ✓ 简单易用
- ✓ 输出清晰
- ✓ 支持带/不带 effort 数据

**使用示例**:

```bash
# 基本测试（不带 effort）
uv run scripts/test_policy_simple.py

# 测试带 effort 的 FAVLA 策略
uv run scripts/test_policy_simple.py --with_effort

# 指定服务器地址和自定义提示
uv run scripts/test_policy_simple.py \
  --host=192.168.1.100 \
  --port=8000 \
  --with_effort \
  --prompt="清理马桶"
```

**参数说明**:
- `--host`: 服务器主机地址（默认: `0.0.0.0`）
- `--port`: 服务器端口（默认: `8000`）
- `--with_effort`: 是否包含力传感器数据
- `--prompt`: 任务提示文本（默认: `"clean the toilet"`）

**输出示例**:
```
正在连接到服务器 0.0.0.0:8000...
✓ 成功连接到服务器
服务器元数据: {...}

发送的观测数据:
  - 图像形状: (224, 224, 3)
  - 状态形状: (7,)
  - Effort 形状: (20, 6)
  - 任务提示: clean the toilet

执行推理...

============================================================
✓ 推理成功！
============================================================
动作输出形状: (1, 50, 32)
动作输出类型: float32

动作统计:
  - 最小值: -2.345678
  - 最大值: 3.456789
  - 平均值: 0.123456
  - 标准差: 1.234567

第一个时间步的前 7 维动作:
  动作[0]: 0.123456
  动作[1]: -0.234567
  ...
```

---

### 2. test_favla_policy.py - 性能测试脚本

**用途**: 测量模型的推理性能，执行多次推理并统计平均时间。

**特点**:
- ✓ 详细的性能统计
- ✓ 支持多步推理
- ✓ 可选的详细输出模式
- ✓ 专门为 FAVLA 策略设计

**使用示例**:

```bash
# 基本性能测试（10 步）
uv run scripts/test_favla_policy.py

# 更多步数以获得更准确的性能数据
uv run scripts/test_favla_policy.py --num_steps=100

# 启用详细输出查看每一步
uv run scripts/test_favla_policy.py --num_steps=5 --verbose

# 指定服务器地址
uv run scripts/test_favla_policy.py \
  --host=192.168.1.100 \
  --port=8000 \
  --num_steps=50
```

**参数说明**:
- `--host`: 服务器主机地址（默认: `0.0.0.0`）
- `--port`: 服务器端口（默认: `8000`）
- `--num_steps`: 执行推理的步数（默认: `10`）
- `--verbose`: 启用详细输出

**输出示例**:
```
2025-10-29 10:30:15 - INFO - 连接到服务器 0.0.0.0:8000...
2025-10-29 10:30:15 - INFO - 服务器元数据: {...}
2025-10-29 10:30:15 - INFO - 发送第一个观测以初始化模型...
2025-10-29 10:30:16 - INFO - 开始执行 10 步推理...

============================================================
测试完成！
总时间: 2.45 秒
平均推理时间: 245.00 毫秒
推理频率: 4.08 Hz
============================================================

最后一次动作输出形状: (1, 50, 32)
最后一次动作输出（前7维）: [0.123, -0.456, 0.789, ...]
```

---

### 3. test_policy_with_dataset.py - 真实数据测试脚本

**用途**: 使用真实的 LeRobot 数据集测试模型，验证在实际数据上的表现。

**特点**:
- ✓ 使用真实数据
- ✓ 测试多个样本
- ✓ 详细的成功/失败统计
- ✓ 支持本地和远程数据集

**使用示例**:

```bash
# 使用本地数据集测试
uv run scripts/test_policy_with_dataset.py \
  --dataset_repo_id=cyx/forceumi1 \
  --num_samples=5

# 从特定索引开始测试
uv run scripts/test_policy_with_dataset.py \
  --dataset_repo_id=cyx/forceumi1 \
  --start_index=100 \
  --num_samples=10

# 使用远程数据集
uv run scripts/test_policy_with_dataset.py \
  --dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
  --local_files_only=False \
  --num_samples=5
```

**参数说明**:
- `--host`: 服务器主机地址（默认: `0.0.0.0`）
- `--port`: 服务器端口（默认: `8000`）
- `--dataset_repo_id`: LeRobot 数据集 repo ID（默认: `cyx/forceumi1`）
- `--local_files_only`: 是否只使用本地文件（默认: `True`）
- `--num_samples`: 测试样本数量（默认: `5`）
- `--start_index`: 从哪个索引开始（默认: `0`）
- `--effort_history_length`: Effort 历史长度（默认: `20`）
- `--default_prompt`: 默认任务提示（默认: `"clean the toilet"`）

**重要说明**:
- 脚本会自动将 PyTorch Tensor 转换为 numpy 数组
- 如果数据集中的 effort 只有单帧，会自动复制为历史序列
- 图像格式会自动从 `[C, H, W]` 转换为 `[H, W, C]`

**输出示例**:
```
============================================================
使用真实数据集测试策略
============================================================

1. 连接到服务器 0.0.0.0:8000...
   ✓ 成功连接
   服务器元数据: {...}

2. 加载数据集 cyx/forceumi1...
   ✓ 数据集加载成功
   数据集大小: 1000 个样本

3. 测试 5 个样本...
------------------------------------------------------------

样本 0:
  观测数据:
    - 图像: (224, 224, 3)
    - 状态: (7,)
    - Effort: (20, 6)
    - 提示: clean the toilet
  ✓ 推理成功
    动作形状: (1, 50, 32)
    前7维: [0.12, -0.34, 0.56, ...]

样本 1:
  ...

============================================================
测试总结
============================================================
成功: 5/5
失败: 0/5

✓ 所有测试通过！
```

---

## 数据格式说明

### 输入观测格式

所有脚本发送的观测数据格式如下：

```python
observation = {
    # 图像数据：字典，包含 "images" 键
    "images": {
        "images": np.ndarray,  # 形状 [224, 224, 3], dtype uint8
    },
    
    # 机器人状态：7 维向量（会被自动填充到 32 维）
    "state": np.ndarray,  # 形状 [7], dtype float32
    
    # 力传感器历史数据（可选，FAVLA 需要）
    "effort": np.ndarray,  # 形状 [20, 6], dtype float32
    
    # 任务描述文本（可选）
    "prompt": str,  # 例如 "clean the toilet"
}
```

### 输出动作格式

服务器返回的动作格式：

```python
action = {
    # 动作序列
    "actions": np.ndarray,  # 形状 [batch_size, action_horizon, action_dim]
                           # 通常是 [1, 50, 32]
}
```

**注意**: 虽然输出是 32 维，但实际使用时只需要前 7 维（对应机器人的 7 个自由度）。

---

## 常见问题

### Q1: "ConnectionRefusedError" 错误

**原因**: 服务器未启动或地址/端口不正确。

**解决方法**:
1. 确认服务器正在运行
2. 检查主机地址和端口是否正确
3. 检查防火墙设置

### Q2: 动作输出全是 NaN

**原因**: 可能是输入数据格式不正确或模型未正确加载。

**解决方法**:
1. 检查输入数据的范围和类型
2. 确认 checkpoint 路径正确
3. 查看服务器日志

### Q3: "ImportError: cannot import name 'LeRobotDataset'"

**原因**: 使用 `test_policy_with_dataset.py` 时需要 lerobot 包。

**解决方法**:
```bash
uv pip install lerobot
```

### Q4: 推理速度慢

**可能原因**:
1. 第一次推理需要 JIT 编译（正常现象）
2. CPU 模式比 GPU 模式慢
3. 网络延迟（如果服务器在远程）

**建议**:
- 使用 `test_favla_policy.py` 测试平均性能
- 第一次推理不计入平均时间
- 考虑使用 GPU 加速

---

## 高级用法

### 自定义观测数据

你可以修改任何脚本中的观测创建函数来使用真实数据：

```python
from PIL import Image
import numpy as np

def create_custom_observation():
    # 加载真实图像
    image = Image.open("path/to/camera_image.jpg")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.uint8)
    
    # 使用真实的机器人状态
    robot_state = get_robot_state()  # 你的函数
    
    # 使用真实的力传感器数据
    force_history = get_force_sensor_history()  # 你的函数
    
    return {
        "images": {"images": image_array},
        "state": robot_state,
        "effort": force_history,
        "prompt": "your task description",
    }
```

### 批量测试

创建一个循环来测试大量样本：

```bash
# 测试数据集的前 1000 个样本，每次测试 100 个
for i in {0..900..100}; do
    uv run scripts/test_policy_with_dataset.py \
        --start_index=$i \
        --num_samples=100
done
```

### 远程服务器测试

如果服务器运行在远程机器上：

```bash
# 替换为实际的服务器 IP
SERVER_IP="192.168.1.100"

uv run scripts/test_policy_simple.py --host=$SERVER_IP
```

---

## 最佳实践

1. **首次测试**: 先用 `test_policy_simple.py` 确认服务器工作正常
2. **性能测试**: 使用 `test_favla_policy.py` 进行至少 50-100 步测试
3. **真实验证**: 使用 `test_policy_with_dataset.py` 在实际数据上验证
4. **持续监控**: 在部署过程中定期运行测试脚本

## 相关文档

- [README_test_favla.md](./README_test_favla.md) - FAVLA 策略详细说明
- [serve_policy.py](./serve_policy.py) - 策略服务器启动脚本
- [OpenPI 文档](../README.md) - 主项目文档

---

**更新日期**: 2025-10-29

