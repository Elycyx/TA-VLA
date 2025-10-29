"""
Effort 历史数据管理器。

用于真机部署时维护力传感器的历史数据缓冲区。

用法示例:
    # 初始化管理器
    effort_manager = EffortHistoryManager(
        effort_dim=6,
        history_offsets=tuple((3*i-60 for i in range(20))),  # FAVLA 配置
        control_frequency=20.0  # 控制频率 20Hz
    )
    
    # 在控制循环中使用
    while True:
        # 读取当前力传感器数据
        current_effort = read_force_sensor()  # 你的函数，返回 [6] 的数组
        
        # 更新历史缓冲区
        effort_manager.update(current_effort)
        
        # 获取用于推理的历史序列
        effort_history = effort_manager.get_history()  # 返回 [20, 6] 的数组
        
        # 构建观测并执行推理
        observation = {
            "images": {...},
            "state": robot_state,
            "effort": effort_history,  # 这里使用历史数据
            "prompt": "clean the toilet"
        }
        action = policy.infer(observation)
"""

from collections import deque
from typing import Sequence

import numpy as np


class EffortHistoryManager:
    """
    管理力传感器历史数据的类。
    
    维护一个滑动窗口缓冲区，根据指定的时间偏移量提取历史数据。
    """
    
    def __init__(
        self,
        effort_dimension: int,
        history_offsets: Sequence[int],
        control_frequency: float = 20.0,
        max_buffer_size: int | None = None,
    ):
        """
        初始化 Effort 历史管理器。
        
        Args:
            effort_dimension: 单帧 effort 的维度（例如 6 维力传感器）
            history_offsets: 历史采样的时间偏移量（相对于当前帧）
                例如: (-60, -57, -54, ..., -3, 0) 表示从 60 帧前开始，每隔 3 帧采样一次
            control_frequency: 控制频率（Hz），用于时间计算
            max_buffer_size: 缓冲区最大大小。如果为 None，自动计算为 abs(min_offset) + 1
        """
        self.effort_dimension = effort_dimension
        self.history_offsets = sorted(history_offsets)  # 确保从小到大排序
        self.control_frequency = control_frequency
        
        # 计算需要的最大缓冲区大小
        min_offset = min(history_offsets)
        if max_buffer_size is None:
            # 需要存储从 min_offset 到当前帧的所有数据
            self.max_buffer_size = abs(min_offset) + 1
        else:
            self.max_buffer_size = max(max_buffer_size, abs(min_offset) + 1)
        
        # 使用 deque 作为循环缓冲区
        self.buffer = deque(maxlen=self.max_buffer_size)
        
        # 初始化为零向量
        self._initialize_buffer()
        
        # 统计信息
        self.total_updates = 0
        
    def _initialize_buffer(self):
        """用零向量初始化缓冲区。"""
        zero_effort = np.zeros(self.effort_dimension, dtype=np.float32)
        for _ in range(self.max_buffer_size):
            self.buffer.append(zero_effort.copy())
    
    def update(self, current_effort: np.ndarray):
        """
        更新缓冲区，添加新的 effort 数据。
        
        Args:
            current_effort: 当前帧的 effort 数据，形状 [effort_dim]
        """
        if current_effort.shape != (self.effort_dimension,):
            raise ValueError(
                f"Effort 数据形状错误: 期望 ({self.effort_dimension},), "
                f"实际 {current_effort.shape}"
            )
        
        # 添加到缓冲区（自动弹出最旧的数据）
        self.buffer.append(current_effort.astype(np.float32).copy())
        self.total_updates += 1
    
    def get_history(self) -> np.ndarray:
        """
        根据历史偏移量获取 effort 历史序列。
        
        Returns:
            形状为 [len(history_offsets), effort_dim] 的数组
        """
        history_list = []
        
        for offset in self.history_offsets:
            # offset 是负数（表示过去）或 0（表示当前）
            # buffer[-1] 是最新的数据，buffer[0] 是最旧的数据
            index = offset  # deque 支持负索引
            
            try:
                history_list.append(self.buffer[index])
            except IndexError:
                # 如果索引超出范围，使用零向量
                history_list.append(np.zeros(self.effort_dimension, dtype=np.float32))
        
        return np.array(history_list, dtype=np.float32)
    
    def is_ready(self) -> bool:
        """
        检查缓冲区是否已准备好（是否已收集足够的历史数据）。
        
        Returns:
            如果已收集足够的历史数据返回 True
        """
        min_offset = min(self.history_offsets)
        return self.total_updates >= abs(min_offset) + 1
    
    def reset(self):
        """重置缓冲区。"""
        self.buffer.clear()
        self._initialize_buffer()
        self.total_updates = 0
    
    def get_buffer_info(self) -> dict:
        """
        获取缓冲区信息。
        
        Returns:
            包含缓冲区状态的字典
        """
        return {
            "effort_dimension": self.effort_dimension,
            "history_length": len(self.history_offsets),
            "history_offsets": self.history_offsets,
            "buffer_size": len(self.buffer),
            "max_buffer_size": self.max_buffer_size,
            "total_updates": self.total_updates,
            "is_ready": self.is_ready(),
            "control_frequency": self.control_frequency,
            "history_duration_seconds": abs(min(self.history_offsets)) / self.control_frequency,
        }


class EffortHistoryManagerWithTimer(EffortHistoryManager):
    """
    带有时间戳的 Effort 历史管理器。
    
    适用于控制频率不稳定的情况，使用实际时间戳进行插值。
    """
    
    def __init__(
        self,
        effort_dimension: int,
        history_offsets: Sequence[int],
        control_frequency: float = 20.0,
        max_buffer_size: int | None = None,
    ):
        super().__init__(effort_dimension, history_offsets, control_frequency, max_buffer_size)
        
        # 存储时间戳（秒）
        self.timestamps = deque(maxlen=self.max_buffer_size)
        
        # 初始化时间戳
        import time
        current_time = time.time()
        for i in range(self.max_buffer_size):
            self.timestamps.append(current_time - (self.max_buffer_size - i) / control_frequency)
    
    def update(self, current_effort: np.ndarray, timestamp: float | None = None):
        """
        更新缓冲区，添加新的 effort 数据和时间戳。
        
        Args:
            current_effort: 当前帧的 effort 数据
            timestamp: 时间戳（秒）。如果为 None，使用当前系统时间
        """
        if timestamp is None:
            import time
            timestamp = time.time()
        
        super().update(current_effort)
        self.timestamps.append(timestamp)
    
    def get_history(self) -> np.ndarray:
        """
        使用时间戳进行插值获取历史数据。
        
        Returns:
            形状为 [len(history_offsets), effort_dim] 的数组
        """
        if len(self.timestamps) < 2:
            # 数据不足，返回零或重复当前值
            return super().get_history()
        
        history_list = []
        current_time = self.timestamps[-1]
        
        for offset in self.history_offsets:
            # 计算目标时间
            target_time = current_time + offset / self.control_frequency
            
            # 线性插值查找最接近的两个时间点
            effort = self._interpolate_effort(target_time)
            history_list.append(effort)
        
        return np.array(history_list, dtype=np.float32)
    
    def _interpolate_effort(self, target_time: float) -> np.ndarray:
        """
        对指定时间的 effort 进行线性插值。
        
        Args:
            target_time: 目标时间戳
        
        Returns:
            插值后的 effort 数据
        """
        timestamps_array = np.array(self.timestamps)
        
        # 如果目标时间早于最早的时间戳，返回最早的数据
        if target_time <= timestamps_array[0]:
            return np.array(self.buffer[0], dtype=np.float32)
        
        # 如果目标时间晚于最晚的时间戳，返回最新的数据
        if target_time >= timestamps_array[-1]:
            return np.array(self.buffer[-1], dtype=np.float32)
        
        # 找到目标时间前后的两个时间点
        idx_after = np.searchsorted(timestamps_array, target_time)
        idx_before = idx_after - 1
        
        time_before = timestamps_array[idx_before]
        time_after = timestamps_array[idx_after]
        effort_before = np.array(self.buffer[idx_before], dtype=np.float32)
        effort_after = np.array(self.buffer[idx_after], dtype=np.float32)
        
        # 线性插值
        alpha = (target_time - time_before) / (time_after - time_before)
        interpolated_effort = effort_before * (1 - alpha) + effort_after * alpha
        
        return interpolated_effort
    
    def reset(self):
        """重置缓冲区和时间戳。"""
        super().reset()
        self.timestamps.clear()
        
        import time
        current_time = time.time()
        for i in range(self.max_buffer_size):
            self.timestamps.append(current_time - (self.max_buffer_size - i) / self.control_frequency)


def create_favla_effort_manager(
    effort_dimension: int = 6,
    control_frequency: float = 20.0,
    use_timer: bool = False,
) -> EffortHistoryManager:
    """
    创建 FAVLA 策略使用的 effort 历史管理器。
    
    Args:
        effort_dimension: Effort 维度（默认 6）
        control_frequency: 控制频率（默认 20 Hz）
        use_timer: 是否使用带时间戳的版本
    
    Returns:
        配置好的 EffortHistoryManager 实例
    """
    # FAVLA 配置: 采样 20 帧，从 -60 到 0，间隔 3 帧
    history_offsets = tuple(3 * i - 60 for i in range(20))
    
    if use_timer:
        return EffortHistoryManagerWithTimer(
            effort_dimension=effort_dimension,
            history_offsets=history_offsets,
            control_frequency=control_frequency,
        )
    else:
        return EffortHistoryManager(
            effort_dimension=effort_dimension,
            history_offsets=history_offsets,
            control_frequency=control_frequency,
        )


# 示例使用
if __name__ == "__main__":
    # 创建 FAVLA effort 管理器
    effort_manager = create_favla_effort_manager(
        effort_dimension=6,
        control_frequency=20.0,
    )
    
    print("Effort 历史管理器信息:")
    print("-" * 60)
    info = effort_manager.get_buffer_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n模拟控制循环:")
    print("-" * 60)
    
    # 模拟 100 步控制循环
    for step in range(100):
        # 模拟读取力传感器数据
        current_effort = np.random.randn(6).astype(np.float32)
        
        # 更新历史
        effort_manager.update(current_effort)
        
        # 获取历史序列
        if effort_manager.is_ready():
            effort_history = effort_manager.get_history()
            
            if step % 20 == 0:  # 每 20 步打印一次
                print(f"步骤 {step}:")
                print(f"  当前 effort: {current_effort}")
                print(f"  历史形状: {effort_history.shape}")
                print(f"  最早帧: {effort_history[0]}")
                print(f"  最新帧: {effort_history[-1]}")
                print()

