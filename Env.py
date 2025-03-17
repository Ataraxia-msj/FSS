import numpy as np
import pandas as pd

# 处理jobtypes.xlsx文件
def load_jobtypes(): 
    # 加载 jobTypes.xlsx 文件
    # df_jobtypes = pd.read_excel("dataset/example_jobtypes.xlsx")
    df_jobtypes = pd.read_excel("dataset/jobTypes.xlsx")
    # 解析 operationTypeSequence 字段，将字符串转换为整数列表
    df_jobtypes["operationTypeSequence"] = df_jobtypes["operationTypeSequence"].apply(lambda x: list(map(int, str(x).split(','))))
    # 转换为字典：key = jobTypeId, value = operationTypeSequence
    job_types = df_jobtypes.set_index("jobTypeName")["operationTypeSequence"].to_dict()
    return job_types

def load_operationtypes():
    # 加载operationTypes.xlsx 文件
    df_operationtypes = pd.read_excel("dataset/operationTypes.xlsx")
    # 转换为字典：key = operationTypeId, value = (operationTypeName, machineTypeId, processingTime)
    operation_types = df_operationtypes.set_index("operationTypeId")[["operationTypeName", "machineTypeId", "processingTime"]].to_dict(orient="index")
    return operation_types


def load_problem():
    # 加载problem.xlsx文件
    df_problem = pd.read_excel("dataset/problem.xlsx")
    def parse_production_requirements(req_str):
        """
        将形如 "A_2,B_1" 的字符串解析成列表 [('A', 2), ('B', 1)]。
        """
        result = []
        for item in req_str.split(','):
            parts = item.split('_')
            if len(parts) == 2:
                try:
                    # 尝试将第二部分转换为整数
                    result.append((parts[0], int(parts[1])))
                except ValueError:
                    result.append((parts[0], parts[1]))
            else:
                result.append(tuple(parts))
        return result
    
    def parse_machine_setting(setting_str):
        """
        将形如 "DARES01_A_DA1,WBRES01_A_WB1" 的字符串解析成列表 
        [('DARES01', 'A_DA1'), ('WBRES01', 'A_WB1')]。
        """
        result = []
        for item in setting_str.split(','):
            parts = item.split('_', 1)  # 只在第一个下划线处分割
            if len(parts) == 2:
                result.append((parts[0], parts[1]))
            else:
                result.append((item,))
        return result
    
    # 对每一行进行转换，返回转换后的字典列表
    converted_data = []
    for index, row in df_problem.iterrows():
        row_dict = {
            'dataset_index': row['dataset_index'],
            'problem_index': row['problem_index'],
            'ProductionRequirements': parse_production_requirements(row['ProductionRequirements']),
            'MachineSetting': parse_machine_setting(row['MachineSetting'])
        }
        converted_data.append(row_dict)
        
    return converted_data

# 操作类型ID与名称对应表
def get_operation_type_name():
    df = pd.read_excel("dataset/operationTypes.xlsx")
    id_name_dict = dict(zip(df[df.columns[0]], df[df.columns[1]]))
    return id_name_dict

def get_setup_time(machine_type_id, is_job_type_same, is_operation_type_same):
    setup_time_lookup = {
    (1, True,  True):  0,
    (1, True,  False): 30,
    (1, False, True):  60,
    (1, False, False): 60,
    (2, True,  True):  61,
    (2, True,  False): 62,
    (2, False, True):  63,
    (2, False, False): 64,
}
    return setup_time_lookup[(machine_type_id, is_job_type_same, is_operation_type_same)]

# 判断目前能进行哪些operation
def get_executable_operations(jobs):
    executable_ops = []
    for job in jobs:
        # 获取作业的操作并按 operation ID 排序
        operations = sorted(jobs[job].items())
        for op, demand in operations:
            if demand > 0:
                # 找到第一个需求量大于 0 的操作
                executable_ops.append(op)
                break
    return executable_ops


class SemiconductorEnv:
    def __init__(self):
        # 加载数据
        self.job_types = load_jobtypes()
        self.operation_types = load_operationtypes()
        self.machine_types = {1: 'DARES', 2: 'WBRES'}
        self.problem_info = load_problem()
        self.machine_status = {}
        self.operation_type_name = get_operation_type_name()
        # print('初始化需求成功，当前产品名称，及其需求量为：',self.problem_info[0]['ProductionRequirements'])

        # 初始化操作类型列表，确保状态向量顺序一致
        self.operation_types_list = list(self.operation_types.keys())
        self.num_operation_types = len(self.operation_types_list)

        # 初始化机器状态
        self.init_machinestaus()
        # print("初始化机器状态成功，当前机器状态为：", self.machine_status)

        # 初始化作业和操作需求
        jobrequirements = self.problem_info[0]['ProductionRequirements']  # [('A', 2), ('B', 1)]
        operationrequirements = {}
        for job, num in jobrequirements:
            for operation in self.job_types[job]:
                if job not in operationrequirements:
                    operationrequirements[job] = {}
                operationrequirements[job][operation] = num
        self.operationrequirements = operationrequirements
        self.jobrequirements = jobrequirements

        # 当前可执行的operation
        self.executable_operations = get_executable_operations(operationrequirements)

        # 时间戳和机器完成时间
        self.timestamp = 0
        self.machine_completion_times = {machine: 0 for machine in self.machine_status}

    # 初始化机器状态   
    def init_machinestaus(self):
        machinesetting = self.problem_info[0]['MachineSetting']
        for machine, setting in machinesetting:
            self.machine_status[machine] = {
                'setting': setting,
                'working': False  # 默认为不工作状态
            }
        
    def state(self):
        """状态由三个 N_O 维向量组成：等待操作数量、空闲机器数量、处理中操作数量"""
        # 向量1：等待操作数量
        waiting_ops = np.zeros(self.num_operation_types)
        for job in self.operationrequirements:
            for op, demand in self.operationrequirements[job].items():
                if op in self.executable_operations and demand > 0:
                    idx = self.operation_types_list.index(op)
                    waiting_ops[idx] = demand

        # 向量2：空闲机器数量
        idle_machines = np.zeros(self.num_operation_types)
        for machine, status in self.machine_status.items():
            if not status['working'] and self.timestamp >= self.machine_completion_times[machine]:
                op_type = self.operation_types_list[[i for i, op in enumerate(self.operation_types_list) if self.operation_type_name[op] in status['setting']][0]]
                idx = self.operation_types_list.index(op_type)
                idle_machines[idx] += 1

        # 向量3：处理中操作数量
        in_process_ops = np.zeros(self.num_operation_types)
        for machine, status in self.machine_status.items():
            if status['working'] and self.timestamp < self.machine_completion_times[machine]:
                op = status['current_operation']
                idx = self.operation_types_list.index(op)
                in_process_ops[idx] += 1

        # 归一化
        total_jobs = sum(num for _, num in self.jobrequirements)
        total_machines = len(self.machine_status)
        state1 = waiting_ops / total_jobs if total_jobs > 0 else waiting_ops
        state2 = idle_machines / total_machines if total_machines > 0 else idle_machines
        state3 = in_process_ops / total_machines if total_machines > 0 else in_process_ops

        state = np.concatenate([state1, state2, state3])
        # print("当前状态为：", state)
        return state
    
    def get_available_actions(self, state):
        """获取可用动作列表"""
        # 生成候选动作
        action_machine_pairs = []
        executable_ops = self.executable_operations # 当前可执行的操作

        for op in executable_ops: # 遍历可执行操作
            job = next(job for job, ops in self.operationrequirements.items() if op in ops) # 获取操作所属的作业
            for machine, status in self.machine_status.items(): # 遍历机器
                if not status['working'] and self.timestamp >= self.machine_completion_times[machine]: # 机器空闲
                    # 检查机器是否能处理该操作
                    machine_setting = status['setting']
                    op_name = self.operation_type_name[op]
                    if op_name in machine_setting:
                        # 计算设置时间
                        machine_type_id = 1 if machine.startswith("DARES") else 2 
                        is_job_type_same = machine_setting.startswith(job)
                        is_operation_type_same = op_name.split('_')[1] in machine_setting
                        setup_time = get_setup_time(machine_type_id, is_job_type_same, is_operation_type_same)

                        # 处理时间
                        processing_time = self.operation_types[op]['processingTime']

                        # 剩余操作数量和时间
                        total_ops = len(self.job_types[job])
                        current_idx = self.job_types[job].index(op)
                        left_operations = total_ops - current_idx - 1
                        left_time = sum(self.operation_types[self.job_types[job][i]]['processingTime'] for i in range(current_idx + 1, total_ops))

                        action_machine_pairs.append((setup_time, processing_time, left_operations, left_time, op, machine))
        return action_machine_pairs

    def calculate_reward(self, prev_time, next_time, setup_time):
        """奖励 = -(切换时间 + 所有机器的空闲时间之和)"""
        idle_time_sum = 0
        for machine, completion_time in self.machine_completion_times.items():
            if not self.machine_status[machine]['working'] or self.timestamp >= completion_time:
                idle_time = next_time - max(prev_time, completion_time)
                idle_time_sum += max(0, idle_time)
        reward = -(setup_time + idle_time_sum)
        # print(f"奖励计算：切换时间={setup_time}, 空闲时间和={idle_time_sum}, 奖励={reward}")
        return reward

    def is_done(self):
        """检查所有操作需求是否为0"""
        for job in self.operationrequirements:
            for op, demand in self.operationrequirements[job].items():
                if demand > 0:
                    return False
        return True

    def reset(self):
        """重置环境"""
        self.operation_log = []  # 初始化操作日志
        self.init_machinestaus()
        jobrequirements = self.problem_info[0]['ProductionRequirements']
        operationrequirements = {}
        for job, num in jobrequirements:
            for operation in self.job_types[job]:
                if job not in operationrequirements:
                    operationrequirements[job] = {}
                operationrequirements[job][operation] = num
        self.operationrequirements = operationrequirements
        self.jobrequirements = jobrequirements
        self.executable_operations = get_executable_operations(self.operationrequirements)
        self.timestamp = 0
        self.machine_completion_times = {machine: 0 for machine in self.machine_status}
        return self.state()

    def step(self, action):
        """执行一步，返回新的状态、奖励和是否结束"""
        # 记录当前时间戳 τ(s_t)
        prev_time = self.timestamp

        # -------------------------------
        # 生成候选动作（融合原 execute_action 逻辑）
        action_machine_pairs = []
        executable_ops = self.executable_operations  # 当前可执行操作

        for op in executable_ops:  # 遍历可执行操作
            # 获取该操作所属的作业
            job = next(job for job, ops in self.operationrequirements.items() if op in ops)
            for machine, status in self.machine_status.items():  # 遍历所有机器
                # 机器必须空闲且当前时间大于等于该机器的完成时间
                if not status['working'] and self.timestamp >= self.machine_completion_times[machine]:
                    # 检查机器是否能处理该操作：比较机器前两字符与操作类型
                    Machinetype = machine[:2]
                    machine_setting = status['setting']
                    op_name = self.operation_type_name[op]
                    op_type = op_name.split('_')[1][:2]
                    if op_type in Machinetype:
                        # 计算设置时间
                        machine_type_id = 1 if machine.startswith("DARES") else 2
                        is_job_type_same = machine_setting.startswith(job)
                        is_operation_type_same = op_name.split('_')[1] in machine_setting
                        setup_time = get_setup_time(machine_type_id, is_job_type_same, is_operation_type_same)

                        # 获取处理时间
                        processing_time = self.operation_types[op]['processingTime']

                        # 计算该作业剩余操作数量和剩余时间
                        total_ops = len(self.job_types[job])
                        current_idx = self.job_types[job].index(op)
                        left_operations = total_ops - current_idx - 1
                        left_time = sum(
                            self.operation_types[self.job_types[job][i]]['processingTime']
                            for i in range(current_idx + 1, total_ops)
                        )

                        action_machine_pairs.append(
                            (setup_time, processing_time, left_operations, left_time, op, machine)
                        )

        # 若没有候选动作，则说明当前时间无法执行任何动作，直接跳转到下一个事件时间
        if not action_machine_pairs:
            # 推进到下一个事件（所有机器中最早的完成时间）
            next_times = [t for t in self.machine_completion_times.values() if t > self.timestamp]
            next_time = min(next_times) if next_times else float('inf')

            # 计算空闲时间：对于每台机器，如果在 [prev_time, next_time] 内处于空闲，则累计空闲时间
            idle_time_sum = 0
            if next_time > self.timestamp:
                for machine, status in self.machine_status.items():
                    comp_time = self.machine_completion_times[machine]
                    if not status['working'] or comp_time <= self.timestamp:
                        idle_time_sum += next_time - self.timestamp
                    elif self.timestamp < comp_time <= next_time:
                        idle_time_sum += next_time - comp_time
                    # 若 comp_time > next_time，则机器在整个区间内均处于工作状态，空闲时间为 0

            # 本次没有执行任何动作，无设置时间，奖励仅为空闲时间的惩罚
            reward = -idle_time_sum

            # 更新时间戳到下一个事件时刻
            self.timestamp = next_time

            # 更新机器状态：若在当前时间戳机器已完成作业，则标记为空闲
            if self.timestamp < float('inf'):
                for machine, comp_time in self.machine_completion_times.items():
                    if self.timestamp >= comp_time and self.machine_status[machine]['working']:
                        self.machine_status[machine]['working'] = False
                        self.machine_status[machine]['current_operation'] = None

            new_state = self.state()
            done = self.is_done()
            return new_state, reward, done, {}

        # 如果存在候选动作，则选择最相似的动作（仅比较前四个元素的欧氏距离）
        def distance(a1, a2):
            return np.sqrt(sum((x - y) ** 2 for x, y in zip(a1[:4], a2[:4])))
        
        selected_action = min(action_machine_pairs, key=lambda x: distance(action, x))
        setup_time, processing_time, _, _, selected_op, selected_machine = selected_action
        job = next(job for job, ops in self.operationrequirements.items() if selected_op in ops)
        
        # print(selected_action)
        # -------------------------------
        
        # 计算该动作执行后的完成时间
        completion_time = self.timestamp + setup_time + processing_time

        # 更新被选中机器状态：标记为工作中，记录当前操作和完成时间
        self.machine_status[selected_machine]['working'] = True
        self.machine_status[selected_machine]['current_operation'] = selected_op
        self.machine_completion_times[selected_machine] = completion_time

        # 更新操作需求：减少对应作业中该操作的剩余需求
        self.operationrequirements[job][selected_op] -= 1
        if self.operationrequirements[job][selected_op] == 0:
            # 若操作需求用尽，则重新计算可执行操作
            self.executable_operations = get_executable_operations(self.operationrequirements)

        # 确定下一个时间戳 τ(s_{t+1})
        available_actions = self.get_available_actions(self.state())
        if available_actions:
            # 如果当前状态下仍有可执行动作，则保持时间不变
            next_time = self.timestamp
        else:
            # 否则推进到下一个事件时间（所有机器中最早的完成时间）
            next_times = [t for t in self.machine_completion_times.values() if t > self.timestamp]
            next_time = min(next_times) if next_times else float('inf')

        # 计算奖励：包含设置时间和从prev_time到next_time期间所有机器的空闲时间
        idle_time_sum = 0
        if next_time > prev_time:
            for machine, status in self.machine_status.items():
                comp_time = self.machine_completion_times[machine]
                if not status['working'] or comp_time <= prev_time:
                    idle_time_sum += next_time - prev_time
                elif prev_time < comp_time <= next_time:
                    idle_time_sum += next_time - comp_time
                # 如果 comp_time > next_time，机器始终处于工作状态，空闲时间为 0
        reward = -(setup_time + idle_time_sum)

        # 更新时间戳到 τ(s_{t+1})
        self.timestamp = next_time

        # 更新所有机器状态：若在当前时间戳机器完成作业，则标记为空闲
        if self.timestamp < float('inf'):
            for machine, comp_time in self.machine_completion_times.items():
                if self.timestamp >= comp_time and self.machine_status[machine]['working']:
                    self.machine_status[machine]['working'] = False
                    self.machine_status[machine]['current_operation'] = None

        new_state = self.state()
        done = self.is_done()

        return new_state, reward, done, {}
        