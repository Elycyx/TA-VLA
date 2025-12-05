"""
使用真实数据集测试策略的脚本。

该脚本从 LeRobot 数据集加载真实数据来测试策略服务器。

用法:
    # 使用数据集测试
    uv run scripts/test_policy_with_dataset.py \\
        --dataset_repo_id=cyx/forceumi1 \\
        --num_samples=5
"""

import dataclasses
import logging
from pathlib import Path

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


@dataclasses.dataclass
class Arguments:
    """使用数据集测试的参数。"""
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 数据集配置
    dataset_repo_id: str = "cyx/forceumi1"
    # 是否使用本地数据集
    local_files_only: bool = True
    # 测试样本数量
    num_samples: int = 5
    # 从哪个索引开始
    start_index: int = 0
    
    # Effort 历史长度（FAVLA 使用 20）
    effort_history_length: int = 20
    
    # 任务提示（如果数据集中没有）
    default_prompt: str = "clean the toilet"


def load_lerobot_dataset(repo_id: str, local_files_only: bool = True):
    """
    加载 LeRobot 数据集。
    
    Args:
        repo_id: 数据集的 repo ID
        local_files_only: 是否只使用本地文件
    
    Returns:
        LeRobot 数据集对象
    """
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        dataset = LeRobotDataset(
            repo_id=repo_id,
            local_files_only=local_files_only,
        )
        return dataset
    except ImportError:
        print("错误: 需要安装 lerobot 包")
        print("请运行: uv pip install lerobot")
        raise
    except Exception as exception:
        print(f"加载数据集时出错: {exception}")
        raise


def convert_to_numpy(data):
    """
    将 PyTorch Tensor 或其他格式转换为 numpy 数组。
    
    Args:
        data: 输入数据（可能是 Tensor、numpy 数组或其他）
    
    Returns:
        numpy 数组
    """
    # 检查是否是 PyTorch Tensor
    if hasattr(data, 'cpu'):
        # PyTorch Tensor
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        # 尝试转换为 numpy 数组
        return np.array(data)


def extract_observation_from_dataset_item(
    item: dict, 
    default_prompt: str = "do something",
    effort_history_length: int = 20,
) -> dict:
    """
    从数据集项中提取观测数据。
    
    Args:
        item: 数据集中的一项
        default_prompt: 默认任务提示
        effort_history_length: Effort 历史长度（默认 20）
    
    Returns:
        观测数据字典
    """
    observation = {}
    
    # 提取图像
    if "observation.images" in item:
        # 转换为 numpy 数组
        image = convert_to_numpy(item["observation.images"])
        
        # 格式可能是 [C, H, W] 或 [H, W, C]
        if image.ndim == 3 and image.shape[0] == 3:
            # [C, H, W] -> [H, W, C]
            image = np.transpose(image, (1, 2, 0))
        
        # 确保是 uint8 格式
        if image.dtype != np.uint8:
            # 假设是 [0, 1] 范围的浮点数
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        observation["images"] = {"images": image}
    
    # 提取状态
    if "observation.state" in item:
        state = convert_to_numpy(item["observation.state"])
        observation["state"] = state.astype(np.float32)
    
    # 提取 effort 数据
    # 注意：LeRobot 数据集中通常只有当前帧的 effort
    # 我们需要创建一个假的历史序列或者重复当前值
    if "observation.effort" in item:
        effort = convert_to_numpy(item["observation.effort"])
        effort = effort.astype(np.float32)
        
        # 如果 effort 只是一维的（单帧），需要扩展为历史序列
        if effort.ndim == 1:
            # 重复当前帧以创建历史序列 [history_length, effort_dim]
            effort = np.tile(effort[np.newaxis, :], (effort_history_length, 1))
        
        observation["effort"] = effort
    
    # 提取提示
    if "prompt" in item and item["prompt"]:
        prompt_value = item["prompt"]
        # 如果 prompt 是 bytes，转换为字符串
        if isinstance(prompt_value, bytes):
            observation["prompt"] = prompt_value.decode('utf-8')
        else:
            observation["prompt"] = str(prompt_value)
    else:
        observation["prompt"] = default_prompt
    
    return observation


def main(arguments: Arguments) -> None:
    """
    主测试函数。
    
    Args:
        arguments: 命令行参数
    """
    print("=" * 60)
    print("使用真实数据集测试策略")
    print("=" * 60)
    
    # 连接到服务器
    print(f"\n1. 连接到服务器 {arguments.host}:{arguments.port}...")
    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=arguments.host,
            port=arguments.port,
        )
        print(f"   ✓ 成功连接")
        print(f"   服务器元数据: {policy.get_server_metadata()}")
    except ConnectionRefusedError:
        print("   ✗ 无法连接到服务器")
        print("\n请先启动服务器:")
        print("  uv run scripts/serve_policy.py policy:checkpoint \\")
        print(f"    --policy.config=pi0_lora_favla \\")
        print("    --policy.dir=<checkpoint路径>")
        return
    
    # 加载数据集
    print(f"\n2. 加载数据集 {arguments.dataset_repo_id}...")
    try:
        dataset = load_lerobot_dataset(
            arguments.dataset_repo_id,
            arguments.local_files_only
        )
        print(f"   ✓ 数据集加载成功")
        print(f"   数据集大小: {len(dataset)} 个样本")
    except Exception as exception:
        print(f"   ✗ 加载数据集失败: {exception}")
        return
    
    # 测试多个样本
    print(f"\n3. 测试 {arguments.num_samples} 个样本...")
    print("-" * 60)
    
    successful_inferences = 0
    failed_inferences = 0
    
    for i in range(arguments.num_samples):
        sample_index = arguments.start_index + i
        
        if sample_index >= len(dataset):
            print(f"\n样本 {sample_index} 超出数据集范围")
            break
        
        print(f"\n样本 {sample_index}:")
        
        try:
            # 获取数据集项
            item = dataset[sample_index]
            
            # 提取观测数据
            observation = extract_observation_from_dataset_item(
                item, 
                default_prompt=arguments.default_prompt,
                effort_history_length=arguments.effort_history_length,
            )
            
            # 显示观测信息
            print(f"  观测数据:")
            if "images" in observation:
                print(f"    - 图像: {observation['images']['images'].shape}")
            if "state" in observation:
                print(f"    - 状态: {observation['state'].shape}")
            if "effort" in observation:
                print(f"    - Effort: {observation['effort'].shape}")
            print(f"    - 提示: {observation['prompt']}")
            
            # 提取数据集中的原始 action
            ground_truth_action = None
            if "action" in item:
                ground_truth_action = convert_to_numpy(item["action"])
                print(f"\n  数据集原始动作:")
                print(f"    形状: {ground_truth_action.shape}")
                # 显示前7维（通常是机器人关节）
                if ground_truth_action.ndim == 1 and len(ground_truth_action) >= 7:
                    print(f"    前7维: {ground_truth_action[:7]}")
                elif ground_truth_action.ndim == 1:
                    print(f"    数据: {ground_truth_action}")
            
            # 执行推理
            action = policy.infer(observation)
            
            # 显示结果
            print(f"\n  ✓ 推理成功")
            actions_array = action['actions']
            print(f"  Policy 输出动作:")
            print(f"    形状: {actions_array.shape}")
            
            # 根据形状灵活显示动作
            # 可能的形状: (action_horizon, action_dim) 或 (batch, action_horizon, action_dim)
            predicted_first_action = None
            if actions_array.ndim == 3:
                # 形状 [batch, action_horizon, action_dim]
                predicted_first_action = actions_array[0, 0]
                print(f"    第一个时间步（前7维）: {predicted_first_action[:7]}")
            elif actions_array.ndim == 2:
                # 形状 [action_horizon, action_dim]
                predicted_first_action = actions_array[0]
                display_action = actions_array[0, :7] if actions_array.shape[1] >= 7 else actions_array[0]
                print(f"    第一个时间步（前7维）: {display_action}")
            else:
                print(f"    数据: {actions_array}")
            
            # 如果有原始动作，计算误差
            if ground_truth_action is not None and predicted_first_action is not None:
                print(f"\n  对比分析:")
                # 计算均方误差 (MSE)
                mse = np.mean((predicted_first_action[:len(ground_truth_action)] - ground_truth_action) ** 2)
                print(f"    MSE (均方误差): {mse:.6f}")
                
                # 计算 L2 距离
                l2_distance = np.linalg.norm(predicted_first_action[:len(ground_truth_action)] - ground_truth_action)
                print(f"    L2 距离: {l2_distance:.6f}")
                
                # 计算每个维度的绝对误差
                abs_errors = np.abs(predicted_first_action[:len(ground_truth_action)] - ground_truth_action)
                print(f"    各维度绝对误差: {abs_errors[:7] if len(abs_errors) >= 7 else abs_errors}")
                print(f"    平均绝对误差: {np.mean(abs_errors):.6f}")
                print(f"    最大绝对误差: {np.max(abs_errors):.6f}")
            
            successful_inferences += 1
            
        except Exception as exception:
            print(f"  ✗ 推理失败: {exception}")
            failed_inferences += 1
            import traceback
            traceback.print_exc()
    
    # 显示总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"成功: {successful_inferences}/{arguments.num_samples}")
    print(f"失败: {failed_inferences}/{arguments.num_samples}")
    
    if successful_inferences == arguments.num_samples:
        print("\n✓ 所有测试通过！")
    elif successful_inferences > 0:
        print(f"\n⚠ 部分测试通过 ({successful_inferences}/{arguments.num_samples})")
    else:
        print("\n✗ 所有测试失败")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s"
    )
    main(tyro.cli(Arguments))

