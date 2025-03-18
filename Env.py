import numpy as np
import pandas as pd

# 处理jobtypes.xlsx文件
def load_jobtypes(): 
    # 加载 jobTypes.xlsx 文件
    df_jobtypes = pd.read_excel("dataset/example_jobtypes.xlsx")
    # df_jobtypes = pd.read_excel("dataset/jobTypes.xlsx")
    # 解析 operationTypeSequence 字段，将字符串转换为整数列表
    df_jobtypes["operationTypeSequence"] = df_jobtypes["operationTypeSequence"].apply(lambda x: list(map(int, str(x).split(','))))
    # 转换为字典：key = jobTypeId, value = operationTypeSequence
    job_types = df_jobtypes.set_index("jobTypeName")["operationTypeSequence"].to_dict()
    return job_types

def load_operationtypes():
    # 加载operationTypes.xlsx 文件
    # df_operationtypes = pd.read_excel("dataset/operationTypes.xlsx")
    df_operationtypes = pd.read_excel("dataset/example_operationtypes.xlsx")
    # 转换为字典：key = operationTypeId, value = (operationTypeName, machineTypeId, processingTime)
    operation_types = df_operationtypes.set_index("operationTypeId")[["operationTypeName", "machineTypeId", "processingTime"]].to_dict(orient="index")
    return operation_types


def load_problem():
    # 加载problem.xlsx文件
    # df_problem = pd.read_excel("dataset/problem.xlsx")
    df_problem = pd.read_excel("dataset/example_problem.xlsx")
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
    (1, True,  False): 3,
    (1, False, True):  6,
    (1, False, False): 6,
    (2, True,  True):  0,
    (2, True,  False): 6.2,
    (2, False, True):  6.3,
    (2, False, False): 6.4,
}
    return setup_time_lookup[(machine_type_id, is_job_type_same, is_operation_type_same)]

# 判断目前能进行哪些operation
def get_executable_operations(jobs_dict, executed):
    # 提取每个 job 的 operation 序列（按编号排序）
    jobs = {}
    for job_name, ops in jobs_dict.items():
        ops_list = sorted(ops.keys())  # 假设 operation 按编号顺序执行
        jobs[job_name] = ops_list
    
    # 获取每个 operation 的剩余次数
    remaining = {op: jobs_dict[job_name][op] for job_name in jobs_dict for op in jobs_dict[job_name]}
    
    # 判断可执行的 operation
    executable = []
    for job_name in jobs:
        ops = jobs[job_name]
        for i, op in enumerate(ops):
            # 检查剩余次数 > 0
            if remaining[op] > 0:
                # 如果是第一个 operation，或前一个 operation 已执行过
                if i == 0 or executed[job_name].get(ops[i-1], 0) > 0:
                    executable.append(op)
    
    return executable

def waiting_ops(d):
    """
    计算等待操作的数量,格式与operationrequirements相同
    """
    result = {}
    for main_key, sub_dict in d.items():
        # 获取键值对列表
        items = list(sub_dict.items())
        # 创建新字典：第一个保持原值，其余置0
        result[main_key] = {
            k: v if i == 0 else 0 
            for i, (k, v) in enumerate(items)
        }
    return result
        

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
        self.executed_operations = {outer_key: {inner_key: 0 for inner_key in operationrequirements[outer_key]} for outer_key in operationrequirements}
        # 当前可执行的operation
        self.executable_operations = get_executable_operations(self.operationrequirements,self.executed_operations)
        # 初始等待操作的数量等于每个job的第一个操作的数量，其他为0
        self.calculate_waiting_ops = waiting_ops(self.operationrequirements)
        
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
        # 计算当前等待加工的操作数量

    def state(self):
        """状态由三个 N_O 维向量组成：等待操作数量、空闲机器数量、处理中操作数量"""
        # 向量1：等待操作数量
        waiting_ops = np.zeros(self.num_operation_types)
        for job, ops in self.calculate_waiting_ops.items():
            for op, num in ops.items():
                idx = self.operation_types_list.index(op)
                waiting_ops[idx] += num

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
    
    def get_available_actions(self):
        """获取可用动作列表"""
        # 生成候选动作
        action_machine_pairs = []
        executable_ops = self.executable_operations # 当前可执行的操作
        # print("当前可执行的操作：",executable_ops)
        for op in executable_ops: # 遍历可执行操作
            job = next(job for job, ops in self.operationrequirements.items() if op in ops) # 获取操作所属的作业
            for machine, status in self.machine_status.items(): # 遍历机器
                if not status['working'] and self.timestamp >= self.machine_completion_times[machine]: # 机器空闲
                    # 检查机器是否能处理该操作
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

                        # 处理时间
                        processing_time = self.operation_types[op]['processingTime']

                        # 剩余操作数量和时间
                        total_ops = len(self.job_types[job])
                        current_idx = self.job_types[job].index(op)
                        left_operations = total_ops - current_idx - 1
                        left_time = sum(self.operation_types[self.job_types[job][i]]['processingTime'] for i in range(current_idx + 1, total_ops))

                        action_machine_pairs.append(
                            (setup_time, processing_time, left_operations, left_time, op, machine))
        return action_machine_pairs
    

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
        self.executable_operations = get_executable_operations(self.operationrequirements,self.executed_operations)
        self.timestamp = 0
        self.machine_completion_times = {machine: 0 for machine in self.machine_status}
        return self.state()

    def execute_action(self, action):
        """执行一步，返回新的状态、奖励和是否结束"""
        # 记录当前时间戳 τ(s_t)
        prev_time = self.timestamp
        action_machine_pairs = self.get_available_actions()
        print("当前可执行的动作：",action_machine_pairs)
        # 输出当前环境的一些信息
        # 若没有候选动作，则说明当前时间无法执行任何动作，直接跳转到下一个事件时间
        if not action_machine_pairs:
            # print("当前时间无法执行任何动作，直接跳转到下一个事件时间")
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
            # print("下一个事件时间：",self.timestamp)

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
        print("选择的动作：",selected_action)
        job = next(job for job, ops in self.operationrequirements.items() if selected_op in ops)
    
        completion_time = self.timestamp + setup_time + processing_time

        # 更新被选中机器状态：标记为工作中，记录当前操作和完成时间
        self.machine_status[selected_machine]['working'] = True
        self.machine_status[selected_machine]['current_operation'] = selected_op
        self.machine_completion_times[selected_machine] = completion_time
        # 把机器的seting改为当前操作，需要从id对应到名称
        self.machine_status[selected_machine]['setting'] = self.operation_type_name[selected_op]

        # 确定下一个时间戳 τ(s_{t+1})
        available_actions = self.get_available_actions()

        # 更新操作需求：减少对应作业中该操作的剩余需求
        self.operationrequirements[job][selected_op] -= 1
        self.executed_operations[job][selected_op] += 1
        self.executable_operations = get_executable_operations(self.operationrequirements,self.executed_operations)
        
        # 更新等待操作数量：当前操作的数量减 1，下一个操作的数量加 1
        current_idx = self.job_types[job].index(selected_op)
        total_ops = len(self.job_types[job])
        if current_idx < total_ops - 1:
            next_op = self.job_types[job][current_idx + 1]
            self.calculate_waiting_ops[job][selected_op] -= 1
            self.calculate_waiting_ops[job][next_op] += 1

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
        # print("时间：",self.timestamp)

        # 更新所有机器状态：若在当前时间戳机器完成作业，则标记为空闲
        if self.timestamp < float('inf'):
            for machine, comp_time in self.machine_completion_times.items():
                if self.timestamp >= comp_time and self.machine_status[machine]['working']:
                    self.machine_status[machine]['working'] = False
                    self.machine_status[machine]['current_operation'] = None

        new_state = self.state()
        done = self.is_done()
        return new_state, reward, done, {}


# env = SemiconductorEnv()
# env.reset()

# print("初始状态：",env.state())
# print('当前机器的状态：',env.machine_status)
# state, reward, done, _ = env.execute_action((0,2,1,3))

# print("第一次操作后状态：",state)
# print('当前机器的状态：',env.machine_status)
# print("第一次操作的奖励：",reward)
# print("是否结束：",done)
# print("当前可执行的操作：",env.executable_operations)
# print("动作机器对：",env.get_available_actions())
# state, reward, done, _ = env.execute_action((0,2,1,3))

# print("第二次操作的状态：",state)
# print('当前机器的状态：',env.machine_status)
# print("第二次操作的奖励：",reward)
# print("是否结束：",done)

# state, reward, done, _ = env.execute_action((6.3,3,0,0))

# print("第三次操作的状态：",state)
# print('当前机器的状态：',env.machine_status)
# print("第三次操作的奖励：",reward)
# print("是否结束：",done)

# state, reward, done, _ = env.execute_action((6,1,1,4))

# print("第四次操作的状态：",state)
# print('当前机器的状态：',env.machine_status)
# print("第四次操作的奖励：",reward)
# print("是否结束：",done)

# state, reward, done, _ = env.execute_action((0,3,0,0))

# print("第五次操作的状态：",state)
# print('当前机器的状态：',env.machine_status)
# print("第五次操作的奖励：",reward)
# print("是否结束：",done)

# state, reward, done, _ = env.execute_action((0,3,0,0))

# print("第六次操作的状态：",state)
# print('当前机器的状态：',env.machine_status)
# print("第六次操作的奖励：",reward)
# print("是否结束：",done)

# state, reward, done, _ = env.execute_action((0,3,0,0))

# print("第七次操作的状态：",state)
# print('当前机器的状态：',env.machine_status)
# print("第七次操作的奖励：",reward)
# print("是否结束：",done)

