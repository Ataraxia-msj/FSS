import numpy as np
import pandas as pd

##############################################################################
# 数据集文件路径

job_file = "dataset\\example_jobtypes.xlsx"
machine_file = "dataset\\machineTypes.xlsx"
operation_file = "dataset\\example_operationtypes.xlsx"
problem_file = "dataset\\example_problem.xlsx"
setup_file = "dataset\\example_setuptime.xlsx"


##############################################################################

# 创建机器类
class Machine:
    def __init__(self, id,type,name, setting):
        """
        初始化机器
        :param typeid: 机器类型ID
        :param type: 机器类型
        :param name: 机器名称
        :param setting: 机器设置
        :param working: 是否工作
        :param current_operation: 当前操作
        :param setup_time: 设置时间
        :param progress_time: 加工时间
        :param wait_time: 等待时间
        :param completion_time: 完成时间
        """
        self.id = id
        self.type = type
        self.name = name
        self.setting = setting
        self.working = False
        self.current_operation = None
        self.setup_time = 0
        self.progress_time = 0
        self.wait_time = 0
        self.completion_time = self.setup_time + self.progress_time + self.wait_time
    
    def is_idle(self, current_time):
        """
        检查机器是否空闲
        :param current_time: 当前时间戳
        :return: 是否空闲
        """
        return not self.working or current_time >= self.completion_time

    def start_operation(self, operation, setup_time, processing_time, current_time, waiting_time):
        """
        开始一个操作
        :param operation: 当前操作
        :param setup_time: 设置时间
        :param processing_time: 加工时间
        :param current_time: 当前时间戳
        """
        self.working = True
        self.current_operation = operation
        self.completion_time = current_time + setup_time + processing_time + waiting_time

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
        :param completed_operations: 已完成操作
        :param used_operations: 已使用的操作
        """
        self.job_id = job_id
        self.job_name = job_name
        self.operation_sequence = operation_sequence
        self.demand = demand
        self.completed_operations = {op: 0 for op in operation_sequence}
        self.used_operations = {op: 0 for op in operation_sequence}  # 新增：跟踪已使用的操作数量

    def is_completed(self):
        """
        检查作业是否完成
        :return: 是否完成
        """
        # 由于我们现在跟踪的是每个实例，而不是类型，需要修改完成状态的检查逻辑
        return all(self.completed_operations[op] >= self.demand for op in self.operation_sequence)


# 创建Opreation类
class Operation:
    def __init__(self, job_id, operation_id, operation_name, machine_type, processing_time,demand,operation_type_id=None):
        """
        初始化操作
        :param job_id: 所属Job ID
        :param operation_type_id: 操作类型ID
        :param operation_name: 操作名称
        :param machine_type_id: 所需机器类型
        :param processing_time: 加工时间
        :param completed: 是否完成
        :param operation_type_id: 操作类型ID
        """
        self.job_id = job_id
        self.operation_id = operation_id
        self.operation_name = operation_name
        self.machine_type = machine_type
        self.processing_time = processing_time
        self.demand = demand
        self.operation_type_id = operation_type_id
        self.completed = False

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
    

# 创建环境类
class SemiconductorEnv:
    def __init__(self, machines, jobs, operations):
        """
        初始化环境
        :param machines: 机器对象列表
        :param jobs: 作业对象列表
        :param operations: 操作对象列表
        """
        self.machines = machines
        self.jobs = jobs
        self.operations = operations
        self.timestamp = 0

    def step(self, action):
        """
        执行一个动作
        :param action: 动作
        :return: 状态、奖励、是否完成、信息
        """
        # 检查机器是否空闲
        for machine in self.machines:
            if machine.working:
                pass
                # 机器在工作，目前不能进行任何操作
            else:
                pass
                # 机器空闲，可以开始操作

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

##############################################################################
# 函数

def load_machines(machine_file, problem_file):
    machine_types_df = pd.read_excel(machine_file)
    problem_df = pd.read_excel(problem_file)
    machine_settings = problem_df["MachineSetting"].iloc[0].split(",")
    machines = []
    for setting in machine_settings:
        parts = setting.split("_")
        machine_name = parts[0]  # 保留完整的机器名称，如 "DARES01"
        operation_name = "_".join(parts[1:])  # 操作名称为剩余部分，如 "A_DA1"
        
        # 提取基本机器类型（不含数字）用于查找对应的机器类型ID
        base_machine_type = ''.join(filter(str.isalpha, machine_name))
        filtered_df = machine_types_df[machine_types_df["machineTypeName"] == base_machine_type]
        if filtered_df.empty:
            raise ValueError(f"Machine type '{base_machine_type}' not found in machine types.")
        machine_type_id = filtered_df["machineTypeId"].values[0]
        
        machines.append(Machine(id=len(machines) + 1, type=machine_type_id, name=machine_name, setting=operation_name))
    return machines

def load_jobs(job_file, problem_file):
    job_types_df = pd.read_excel(job_file)
    problem_df = pd.read_excel(problem_file)
    production_requirements = problem_df["ProductionRequirements"].iloc[0].split(",")
    jobs = []
    for requirement in production_requirements:
        job_name, demand = requirement.split("_")
        demand = int(demand)
        operation_sequence = job_types_df[job_types_df["jobTypeName"] == job_name]["operationTypeSequence"].values[0]
        operation_sequence = list(map(int, operation_sequence.split(",")))
        jobs.append(Job(job_id=len(jobs) + 1, job_name=job_name, operation_sequence=operation_sequence, demand=demand))
    return jobs

def load_operations(operation_file):
    """
    从操作类型文件中加载操作类型信息
    :param operation_file: 操作类型文件路径
    :return: 操作类型列表
    """
    operation_types_df = pd.read_excel(operation_file)
    operations = []
    for _, row in operation_types_df.iterrows():
        operations.append(Operation(
            job_id=None,  # 初始时不关联到具体作业
            operation_id=row["operationTypeId"],
            operation_name=row["operationTypeName"],
            machine_type=row["machineTypeId"],
            processing_time=row["processingTime"],
            demand=0  # 初始需求量为0，后续会根据作业需求更新
        ))
    return operations

def create_operation_instances(operations, jobs):
    """
    根据作业需求量创建实际的操作实例
    :param operations: 操作类型列表
    :param jobs: 作业列表
    :return: 操作实例列表
    """
    operation_instances = []
    instance_id = 1
    
    # 创建操作类型ID到操作对象的映射
    op_type_map = {op.operation_id: op for op in operations}
    
    for job in jobs:
        for _ in range(job.demand):  # 为每个需求创建一套操作
            for op_type_id in job.operation_sequence:
                # 找到对应的操作类型
                op_type = op_type_map.get(op_type_id)
                if op_type:
                    # 创建新的操作实例
                    instance = Operation(
                        job_id=job.job_id,
                        operation_id=instance_id,  # 使用新的实例ID
                        operation_name=op_type.operation_name,
                        machine_type=op_type.machine_type,
                        processing_time=op_type.processing_time,
                        operation_type_id=op_type_id,
                        demand=1  # 每个实例的需求量为1
                    )
                    operation_instances.append(instance)
                    instance_id += 1
    
    return operation_instances

def get_setup_time(machine_type_id, is_job_type_same, is_operation_type_same):
    """
    获取设置时间
    :param machine_type_id: 机器类型ID
    :param is_job_type_same: 是否相同作业类型
    :param is_operation_type_same: 是否相同操作类型
    :return: 设置时间
    """
    setup_time_lookup = {
        (1, True,  True):  0,
        (1, True,  False): 3,
        (1, False, True):  6,
        (1, False, False): 6,
        (2, True,  True):  0,
        (2, True,  False): 6.2,
        (2, False, True):  6.3,
        (2, False, False): 6.4,
    }
    return setup_time_lookup[(machine_type_id, is_job_type_same, is_operation_type_same)]

# 切换时间戳
def transfrom_time(now_time, machines):
    """
    切换时间戳到下一个机器完成操作的时间点
    
    :param now_time: 当前时间戳
    :param machines: 机器列表
    :return: 新的时间戳
    """ 
    # 找到下一个即将完成操作的机器
    next_completion_time = float('inf')
    completing_machines = []

    for machine in machines:
        if machine.working and machine.setup_time > now_time:
            if machine.completion_time < next_completion_time:
                next_completion_time = machine.completion_time
                completing_machines = [machine]
            elif machine.completion_time == next_completion_time:
                completing_machines.append(machine)
    
    # 如果没有正在工作的机器，返回当前时间
    if next_completion_time == float('inf'):
        return now_time
    
    # 更新所有在该时间点完成操作的机器状态
    for machine in completing_machines:
        machine.finish_operation(next_completion_time)
        
    # 更新时间到下一个完成时间
    return next_completion_time

# 调度操作，还没有写完
def schedule_operation(operation, machine, current_time, jobs):
    """
    安排一个操作到机器上
    
    :param operation: 要安排的操作
    :param machine: 要使用的机器
    :param current_time: 当前时间戳
    :param jobs: 作业列表
    :return: 操作的完成时间
    """
    # 找到操作所属的作业
    job = next((j for j in jobs if j.job_id == operation.job_id), None)
    if not job:
        return current_time
        
    # 获取操作在作业序列中的位置
    op_sequence = job.operation_sequence
    op_index = op_sequence.index(operation.operation_type_id)
    
    # 计算设置时间
    prev_op = machine.current_operation
    is_job_type_same = prev_op and prev_op.job_id == operation.job_id
    is_operation_type_same = prev_op and prev_op.operation_type_id == operation.operation_type_id
    setup_time = get_setup_time(machine.type, is_job_type_same, is_operation_type_same)
    
    # 检查前置操作是否完成
    waiting_time = 0
    if op_index > 0:
        prev_op_type_id = op_sequence[op_index - 1]
        # 检查是否有已完成但尚未被后续操作使用的前置操作实例
        # 需要跟踪已完成但尚未被后续操作使用的前置操作数量
        available_prev_ops = job.completed_operations[prev_op_type_id] - job.used_operations.get(prev_op_type_id, 0)

        if available_prev_ops <= 0:
            # 没有可用的前置操作实例，需要等待
            # waiting_time = estimate_waiting_time(prev_op_type_id, machines, jobs)
            pass
        else:
            # 有可用的前置操作实例，可以立即开始
            # 更新已使用的前置操作数量
            job.used_operations[prev_op_type_id] = job.used_operations.get(prev_op_type_id, 0) + 1
    
    # 开始操作
    machine.start_operation(operation, setup_time, operation.processing_time, current_time, waiting_time)
    return machine.completion_time


    