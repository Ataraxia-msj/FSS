import numpy as np
import pandas as pd

##############################################################################
# 变量

job_file = "dataset\\example_jobtypes.xlsx"
machine_file = "dataset\\machineTypes.xlsx"
operation_file = "dataset\\example_operationtypes.xlsx"
problem_file = "dataset\\example_problem.xlsx"
setup_file = "dataset\\example_setuptime.xlsx"

# 等待执行的operation以及它的数量
waiting_operations = {}

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

    def start_operation(self, operation, setup_time, processing_time, waiting_time):
        """
        开始一个操作
        :param operation: 当前操作
        :param setup_time: 设置时间
        :param processing_time: 加工时间
        :param waiting_time: 等待时间
        """
        self.working = True
        self.current_operation = operation
        self.completion_time = self.completion_time + setup_time + processing_time + waiting_time

    def finish_operation(self):
        """
        完成当前操作
        :param current_time: 当前时间戳
        """
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
    def __init__(self, job_id, operation_id, operation_name, machine_type, processing_time, demand, operation_type_id=None):
        self.job_id = job_id
        self.operation_id = operation_id
        self.operation_name = operation_name
        self.machine_type = machine_type
        self.processing_time = processing_time
        self.demand = demand
        self.completed = False
        self.operation_type_id = operation_type_id
        
        # 新增时间相关字段
        self.start_time = None
        self.completion_time = None
        
        # 前驱和后继操作（将在create_operation_instances中设置）
        self.predecessor = None  # 前驱操作实例ID
        self.successor = None    # 后继操作实例ID
        
        # 记录操作在作业中的位置
        self.position_in_job = None


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

        # 创建操作实例
        self.operation_instances, self.job_op_sequences = create_operation_instances(self.operations, self.jobs)

        # 然后初始化等待操作
        self.waiting_operations = self.initialize_waiting_operations()
        

    def step(self, action):
        """
        执行一个动作
        :param action: 动作
        :return: 下一个状态、奖励、是否完成
        """
        # 获取可用的动作列表
        available_actions = self.get_available_actions()
        print("可用动作：", available_actions)
        if not available_actions:
            print("没有可用的动作")
            return
        else:
            # 选择动作
            selected_action = self.select_action(action, available_actions)
            print("选择的动作：", selected_action)
            # 执行动作
            settime, waittime = self.execute_action(selected_action)
            print("设置时间：", settime)
            print("等待时间：", waittime)
        reward = -settime - waittime
        # 机器完成加工
        for machine in self.machines:
            if machine.working:
                machine.finish_operation()
        return reward

        


    def reset(self):
        """
        重置环境
        """
        self.timestamp = 0
        for job in self.jobs:
            job.completed_operations = {op: 0 for op in job.operation_sequence}
            job.used_operations = {op: 0 for op in job.operation_sequence}

        for machine in self.machines:
            machine.working = False
            machine.current_operation =  None
            machine.completion_time = 0
        for op in self.operations:
            op.completed = False
        
        waiting_operations = self.initialize_waiting_operations()
    

    def initialize_waiting_operations(self):
        """
        初始化等待操作量字典，只有没有前驱的操作才能被添加
        """
        waiting_ops = {}
        
        # 初始化所有操作的等待量为0
        for op in self.operation_instances:
            waiting_ops[op.operation_id] = 0
        
        # 只将没有前驱的操作添加到等待队列
        for op in self.operation_instances:
            if not op.predecessor:  # 没有前驱操作
                waiting_ops[op.operation_id] = 1
        
        return waiting_ops

    def get_available_actions(self):
        """
        根据等待加工的操作，查找可用的动作集
        :return: 可用的动作集，每个动作是 (操作ID, 机器ID, 设置时间, 加工时间, 剩余操作数量, 剩余操作时间)
        """
        available_actions = []
        
        # 获取等待加工的操作(值为1的操作)
        waiting_ops = [op_id for op_id, count in self.waiting_operations.items() if count > 0]
        
        # 对每个等待操作，找到可以加工它的机器
        for op_id in waiting_ops:
            # 找到对应的操作实例
            op_instance = next((op for op in self.operation_instances if op.operation_id == op_id), None)
            if not op_instance:
                continue
            
            # 查找可以处理此操作类型的空闲机器
            for machine in self.machines:
                if machine.type == op_instance.machine_type and machine.is_idle(self.timestamp):
                    # 计算设置时间
                    setup_time = self.calculate_setup_time(machine, op_instance)
                    
                    # 获取加工时间
                    processing_time = op_instance.processing_time
                    
                    # 计算剩余操作数量和时间
                    remaining_ops_count, remaining_ops_time = self.calculate_remaining_ops(op_instance)
                    
                    # 添加到可用动作集
                    action = (op_id, machine.id, setup_time, processing_time, 
                            remaining_ops_count, remaining_ops_time)
                    available_actions.append(action)
        
        return available_actions
    
    def calculate_setup_time(self, machine, operation):
        """
        计算在给定机器上加工操作所需的设置时间，基于机器当前设置和操作类型
        :param machine: 机器对象
        :param operation: 操作对象
        :return: 设置时间
        """
        # 如果机器处于空闲状态且没有初始设置，使用默认设置时间
        if not machine.setting:
            return 6  # 默认最大设置时间
        
        # 获取操作的作业和操作类型信息
        operation_job = next((j for j in self.jobs if j.job_id == operation.job_id), None)
        if not operation_job:
            return 6
        
        # 解析机器的当前设置
        # 假设设置格式为 "操作名称_作业类型"，例如 "A_DA1"
        setting_parts = machine.setting.split("_")
        if len(setting_parts) < 2:
            return 6  # 格式不符，使用默认值
        
        setting_op_type = machine.setting  # 第一部分是操作类型
        setting_job_type = setting_parts[0]  # 第二部分是作业类型

        # 检查作业类型和操作类型是否相同
        is_job_type_same = setting_job_type == operation_job.job_name
        is_operation_type_same = setting_op_type == operation.operation_name
        
        # 使用查找表获取设置时间
        return get_setup_time(machine.type, is_job_type_same, is_operation_type_same)


    def calculate_remaining_ops(self, operation):
        """
        计算剩余操作的数量和总时间
        :param operation: 当前操作对象
        :return: (剩余操作数量, 剩余操作总时间)
        """
        # 获取操作所属的作业
        job = next((j for j in self.jobs if j.job_id == operation.job_id), None)
        if not job:
            return 0, 0
        
        # 获取该作业的操作序列列表
        job_sequences = self.job_op_sequences.get(job.job_id, [])
        
        # 找到包含当前操作的序列
        for seq in job_sequences:
            op_ids = [op.operation_id for op in seq]
            if operation.operation_id in op_ids:
                # 找到当前操作在序列中的位置
                op_index = op_ids.index(operation.operation_id)
                
                # 计算剩余操作数量和时间
                remaining_ops = seq[op_index+1:]
                remaining_count = len(remaining_ops)
                remaining_time = sum(op.processing_time for op in remaining_ops)
                
                return remaining_count, remaining_time
        
        return 0, 0
    
    def finish_operation(self, current_time, operation=None):
        """
        完成当前操作，并更新机器设置状态
        :param current_time: 当前时间戳
        :param operation: 完成的操作对象
        """
        if current_time >= self.completion_time:
            if operation:
                # 获取操作对应的作业
                job = None
                # 这里需要通过环境查找对应的作业
                # 应由环境类调用并提供作业信息
                
                # 更新机器设置为当前完成的操作
                self.setting = f"{operation.operation_name}_{job.job_name if job else ''}"
            
            self.working = False
            self.current_operation = None

    def select_action(self, action, available_actions):
        """
        选择一个动作
        :param available_actions: 可用的动作集
        :return: 选择的动作
        """
        # 选择action和avaliable_actions中欧几里得距离最近的动作
        # 将action转换为numpy数组（确保是四维向量）
        action_array = np.array(action)
        
        # 提取每个可用动作的后四个参数（设置时间、加工时间、剩余操作数量、剩余操作时间）
        features_array = np.array([a[2:] for a in available_actions])
        
        # 计算欧几里得距离
        distances = np.linalg.norm(features_array - action_array, axis=1)
        
        # 选择距离最小的动作
        selected_index = np.argmin(distances)
        return available_actions[selected_index]
    
    def execute_action(self, action):
        """
        执行一个动作
        :param action: 动作
        """
        # 解析动作
        op_id, machine_id, setup_time, processing_time, remaining_ops_count, remaining_ops_time = action
        
        # 找到操作实例
        operation = next((op for op in self.operation_instances if op.operation_id == op_id), None)
        if not operation:
            return
        
        # 找到机器
        machine = next((m for m in self.machines if m.id == machine_id), None)
        if not machine:
            return
        
        # 更新操作的开始时间,开始时间为该机器当前完成时间
        operation.start_time = machine.completion_time
        
        # 更新机器的设置时间
        machine.setup_time = setup_time
        
        # 计算等待时间
        wait_time = 0
        if operation.predecessor:
            # 找到前驱操作
            predecessor = next((op for op in self.operation_instances if op.operation_id == operation.predecessor), None)
            if predecessor and not predecessor.completed:
                operation_start_with_setup = machine.completion_time + setup_time
                if predecessor.completion_time > operation_start_with_setup:
                    wait_time = predecessor.completion_time - operation_start_with_setup
                else:
                    wait_time = 0

        # 开始操作
        machine.start_operation(operation, setup_time, processing_time, wait_time)
        
        # 更新等待操作量
        self.waiting_operations[op_id] -= 1
        
        # 更新操作的完成时间
        operation.completion_time = machine.completion_time
        
        # 更新操作的完成状态
        operation.completed = True

        # 检查是否有后继操作，将其添加到等待队列
        if operation.successor:
            # 获取后继操作
            successor_op = next((op for op in self.operation_instances if op.operation_id == operation.successor), None)
            if successor_op:
                # 检查所有前置操作是否完成
                predecessors_completed = True
                if successor_op.predecessor:
                    # 确认前置操作已完成
                    predecessor = next((op for op in self.operation_instances if op.operation_id == successor_op.predecessor), None)
                    if predecessor and not predecessor.completed:
                        predecessors_completed = False
                
                # 如果所有前置操作都已完成，则将后继操作添加到等待队列
                if predecessors_completed:
                    self.waiting_operations[operation.successor] = 1
                    print(f"操作 {operation.successor} 已添加到等待队列")
        
        return setup_time,wait_time



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

def load_operations(operation_file, job_file):
    """
    从操作类型文件中加载操作类型信息
    :param operation_file: 操作类型文件路径
    :param job_file: 作业类型文件路径
    :return: 操作类型列表
    """
    operation_types_df = pd.read_excel(operation_file)
    job_types_df = pd.read_excel(job_file)
    operations = []
    
    # 创建操作ID到作业ID的映射
    op_to_job_map = {}
    for _, job_row in job_types_df.iterrows():
        job_id = job_row["jobTypeId"]
        op_sequence = job_row["operationTypeSequence"].split(",")
        for op_id in op_sequence:
            op_id = int(op_id)
            if op_id not in op_to_job_map:
                op_to_job_map[op_id] = []
            op_to_job_map[op_id].append(job_id)
    
    for _, row in operation_types_df.iterrows():
        op_id = row["operationTypeId"]
        # 获取操作所属的作业ID（可能有多个）
        job_ids = op_to_job_map.get(op_id, [])
        # 使用第一个找到的作业ID，如果没有则使用None
        job_id = job_ids[0] if job_ids else None
        
        operations.append(Operation(
            job_id=job_id,
            operation_id=op_id,
            operation_name=row["operationTypeName"],
            machine_type=row["machineTypeId"],
            processing_time=row["processingTime"],
            demand=0  # 初始需求量为0，后续会根据作业需求更新
        ))
    return operations

def create_operation_instances(operations, jobs):
    """
    根据作业需求量创建实际的操作实例，并建立它们之间的联系
    :param operations: 操作类型列表
    :param jobs: 作业列表
    :return: 操作实例列表
    """
    operation_instances = []
    instance_id = 1
    
    # 创建操作类型ID到操作对象的映射
    op_type_map = {op.operation_id: op for op in operations}
    
    # 用于存储每个作业的操作实例序列
    job_op_sequences = {job.job_id: [] for job in jobs}
    
    for job in jobs:
        for demand_index in range(job.demand):  # 为每个需求创建一套操作
            job_sequence = []  # 存储当前作业实例的操作序列
            
            for position, op_type_id in enumerate(job.operation_sequence):
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
                    # 设置操作在作业中的位置
                    instance.position_in_job = position
                    
                    operation_instances.append(instance)
                    job_sequence.append(instance)
                    instance_id += 1
            
            # 为这个作业中的操作建立前驱后继关系
            for i in range(len(job_sequence)):
                if i > 0:  # 不是第一个操作
                    job_sequence[i].predecessor = job_sequence[i-1].operation_id
                if i < len(job_sequence) - 1:  # 不是最后一个操作
                    job_sequence[i].successor = job_sequence[i+1].operation_id
            
            # 保存作业的操作序列
            job_op_sequences[job.job_id].append(job_sequence)
    
    return operation_instances, job_op_sequences

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



##############################################################################
env = SemiconductorEnv(
    machines=load_machines(machine_file, problem_file),
    jobs=load_jobs(job_file, problem_file),
    operations=load_operations(operation_file, job_file)
)
reward = env.step((0,2,1,3))
print("奖励：", reward)

reward = env.step((0,2,1,3))
print("奖励：", reward)

reward = env.step((6,2,1,3))
print("奖励：", reward)

reward = env.step((6,2,1,3))
print("奖励：", reward)
    