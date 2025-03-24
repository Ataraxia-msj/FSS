import numpy as np
import pandas as pd


job_file = "dataset\\example_jobtypes.xlsx"
machine_file = "dataset\\machineTypes.xlsx"
operation_file = "dataset\\example_operationtypes.xlsx"
problem_file = "dataset\\example_problem.xlsx"
setup_file = "dataset\\example_setuptime.xlsx"

# 创建机器类
class Machine:
    def __init__(self, id,type,name, setting):
        """
        初始化机器
        :param typeid: 机器类型ID
        :param type: 机器类型
        :param name: 机器名称
        :param setting: 机器初始设置
        :param working: 是否工作
        :param current_operation: 当前操作
        :param completion_time: 完成时间
        """
        self.id = id
        self.type = type
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

# 加载数据
# 加载基础数据
print("加载机器数据...")
machines = load_machines(machine_file, problem_file)
print(f"成功加载 {len(machines)} 台机器:")
for machine in machines:
    print(f"  ID: {machine.id}, 类型: {machine.type}, 名称: {machine.name}, 设置: {machine.setting}, 空闲: {machine.is_idle(0)}, 完成时间: {machine.completion_time}, 当前操作: {machine.current_operation}")

print("\n加载作业数据...")
jobs = load_jobs(job_file, problem_file)
print(f"成功加载 {len(jobs)} 个作业:")
for job in jobs:
    print(f"  ID: {job.job_id}, 名称: {job.job_name}, 操作序列: {job.operation_sequence}, 需求量: {job.demand}")

print("\n加载操作类型数据...")
operation_types = load_operations(operation_file)
print(f"成功加载 {len(operation_types)} 个操作类型:")
for operation in operation_types:
    print(f"  ID: {operation.operation_id}, 名称: {operation.operation_name}, 机器类型: {operation.machine_type}, 处理时间: {operation.processing_time}")

# 创建操作实例
print("\n创建操作实例...")
operations = create_operation_instances(operation_types, jobs)
print(f"成功创建 {len(operations)} 个操作实例:")
for operation in operations:
    print(f"  ID: {operation.operation_id}, 名称: {operation.operation_name}, 类型ID: {operation.operation_type_id}, 作业ID: {operation.job_id}, 机器类型: {operation.machine_type}, 处理时间: {operation.processing_time}")
