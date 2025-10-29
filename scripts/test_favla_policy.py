"""
测试 FAVLA 策略的脚本。

该脚本创建 dummy 输入数据并发送到正在运行的策略服务器，以测试模型输出。

用法:
    uv run scripts/test_favla_policy.py --host=0.0.0.0 --port=8000 --num_steps=10
"""

import dataclasses
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


@dataclasses.dataclass
class Arguments:
    """测试脚本的参数。"""
    
    # 服务器主机地址
    host: str = "0.0.0.0"
    # 服务器端口
    port: int = 8000
    # 执行推理的步数
    num_steps: int = 10
    # 是否打印详细输出
    verbose: bool = False


def create_dummy_observation_for_favla() -> dict:
    """
    为 FAVLA 策略创建 dummy 观测数据。
    
    根据 FavlaInputs 转换函数的期望，创建包含以下内容的观测数据：
    - images: 包含 "images" 键的字典
    - state: 状态向量（会被填充到 action_dim=32）
    - effort: 力传感器历史数据（如果使用）
    - prompt: 任务提示语（可选，默认为 "clean the toilet"）
    
    Returns:
        包含观测数据的字典
    """
    # 图像尺寸: (高度, 宽度, 通道数) = (224, 224, 3)
    # 使用 uint8 格式，范围 [0, 255]
    image_height = 224
    image_width = 224
    image_channels = 3
    
    # 创建随机图像数据
    random_image = np.random.randint(
        low=0, 
        high=256, 
        size=(image_height, image_width, image_channels), 
        dtype=np.uint8
    )
    
    # 状态维度：原始维度可以是任意的，会被自动填充到 32
    # 这里使用 7 维（对应机器人的 7 个自由度）
    state_dimension = 7
    random_state = np.random.rand(state_dimension).astype(np.float32)
    
    # Effort (力传感器) 历史数据
    # 根据配置: effort_history = tuple((3*i-60 for i in range(20)))
    # 这表示采样 20 帧历史数据，从 -60 到 0，间隔 3 帧
    # effort_dim = 6 (单帧的力传感器维度)
    effort_history_length = 20
    effort_single_frame_dimension = 6
    
    # 创建随机 effort 数据
    # 注意：实际应用中，这些值应该是归一化的力传感器读数
    random_effort = np.random.randn(
        effort_history_length, 
        effort_single_frame_dimension
    ).astype(np.float32)
    
    # 任务提示语
    # 注意：如果配置中设置了 default_prompt，则这个字段是可选的
    task_prompt = "clean the toilet"
    
    observation = {
        "images": {
            "images": random_image,
        },
        "state": random_state,
        "effort": random_effort,
        "prompt": task_prompt,
    }
    
    return observation


def main(arguments: Arguments) -> None:
    """
    主函数：连接到策略服务器并执行推理测试。
    
    Args:
        arguments: 命令行参数
    """
    # 创建 websocket 客户端策略
    logging.info(f"连接到服务器 {arguments.host}:{arguments.port}...")
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=arguments.host,
        port=arguments.port,
    )
    
    # 打印服务器元数据
    server_metadata = policy.get_server_metadata()
    logging.info(f"服务器元数据: {server_metadata}")
    
    # 发送第一个观测，确保模型已加载
    logging.info("发送第一个观测以初始化模型...")
    first_observation = create_dummy_observation_for_favla()
    first_action = policy.infer(first_observation)
    
    if arguments.verbose:
        logging.info(f"第一次推理的动作输出形状: {first_action['actions'].shape}")
        logging.info(f"第一次推理的动作输出: {first_action['actions']}")
    
    # 执行多步推理以测试性能
    logging.info(f"开始执行 {arguments.num_steps} 步推理...")
    start_time = time.time()
    
    for step_index in range(arguments.num_steps):
        observation = create_dummy_observation_for_favla()
        action = policy.infer(observation)
        
        if arguments.verbose:
            logging.info(f"步骤 {step_index + 1}: 动作形状 = {action['actions'].shape}")
            # 只显示前 7 维（实际使用的维度）
            logging.info(f"步骤 {step_index + 1}: 动作[:7] = {action['actions'][0, :7]}")
    
    end_time = time.time()
    
    # 打印性能统计
    total_time = end_time - start_time
    average_time = total_time / arguments.num_steps
    
    print("\n" + "=" * 60)
    print(f"测试完成！")
    print(f"总时间: {total_time:.2f} 秒")
    print(f"平均推理时间: {average_time * 1000:.2f} 毫秒")
    print(f"推理频率: {1.0 / average_time:.2f} Hz")
    print("=" * 60)
    
    # 打印最终动作的详细信息
    if not arguments.verbose:
        print(f"\n最后一次动作输出形状: {action['actions'].shape}")
        print(f"最后一次动作输出（前7维）: {action['actions'][0, :7]}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main(tyro.cli(Arguments))

