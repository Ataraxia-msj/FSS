import numpy as np
import pandas as pd

# 创建机器类
class Machine:
    def __init__(self, name, setting):
        """
        初始化机器
        :param name: 机器名称
        :param setting: 机器初始设置
        """
        self.name = name
        self.setting = setting
        self.working = False
        self.current_operation = None
        self.completion_time = 0

    def is_idle(self, current_time):
        """
        检查机器是否空闲
        :param current_time: 当前时间戳
        :return: 是否空闲
        """
        return not self.working or current_time >= self.completion_time

    def start_operation(self, operation, setup_time, processing_time, current_time):
        """
        开始一个操作
        :param operation: 当前操作
        :param setup_time: 设置时间
        :param processing_time: 加工时间
        :param current_time: 当前时间戳
        """
        self.working = True
        self.current_operation = operation
        self.completion_time = current_time + setup_time + processing_time

    def finish_operation(self, current_time):
        """
        完成当前操作
        :param current_time: 当前时间戳
        """
        if current_time >= self.completion_time:
            self.working = False
            self.current_operation = None



# 创建Job类
class Job:
    def __init__(self, job_id, job_name, operation_sequence, demand):
        """
        初始化作业
        :param job_id: 作业ID
        :param job_name: 作业名称
        :param operation_sequence: 操作序列（列表）
        :param demand: 作业需求量
        """
        self.job_id = job_id
        self.job_name = job_name
        self.operation_sequence = operation_sequence
        self.demand = demand
        self.completed_operations = {op: 0 for op in operation_sequence}

    def is_completed(self):
        """
        检查作业是否完成
        :return: 是否完成
        """
        return all(self.completed_operations[op] >= self.demand for op in self.operation_sequence)


# 创建Opreation类
class Operation:
    def __init__(self, operation_id, operation_name, machine_type, processing_time):
        """
        初始化操作
        :param operation_id: 操作ID
        :param operation_name: 操作名称
        :param machine_type: 所需机器类型
        :param processing_time: 加工时间
        """
        self.operation_id = operation_id
        self.operation_name = operation_name
        self.machine_type = machine_type
        self.processing_time = processing_time


# 创建环境类
class SemiconductorEnv:
    def __init__(self, jobs, machines, operations):
        """
        初始化环境
        :param jobs: 作业列表
        :param machines: 机器列表
        :param operations: 操作列表
        """
        self.jobs = jobs
        self.machines = machines
        self.operations = operations
        self.timestamp = 0

    def reset(self):
        """
        重置环境
        """
        self.timestamp = 0
        for job in self.jobs:
            job.completed_operations = {op: 0 for op in job.operation_sequence}
        for machine in self.machines:
            machine.working = False
            machine.current_operation = None
            machine.completion_time = 0



# 创建State类
class State:
    def __init__(self, waiting_ops, idle_machines, in_process_ops):
        """
        初始化状态
        :param waiting_ops: 等待操作数量
        :param idle_machines: 空闲机器数量
        :param in_process_ops: 正在处理的操作数量
        """
        self.waiting_ops = waiting_ops
        self.idle_machines = idle_machines
        self.in_process_ops = in_process_ops

    def to_vector(self):
        """
        转换为向量表示
        """
        return np.concatenate([self.waiting_ops, self.idle_machines, self.in_process_ops])