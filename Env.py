import numpy as np
import pandas as pd

# 处理jobtypes.xlsx文件
def load_jobtypes(): 
    # 加载 jobTypes.xlsx 文件
    df_jobtypes = pd.read_excel("dataset/example_jobtypes.xlsx")
    # 解析 operationTypeSequence 字段，将字符串转换为整数列表
    df_jobtypes["operationTypeSequence"] = df_jobtypes["operationTypeSequence"].apply(lambda x: list(map(int, str(x).split(','))))
    # 转换为字典：key = jobTypeId, value = operationTypeSequence
    job_types = df_jobtypes.set_index("jobTypeName")["operationTypeSequence"].to_dict()
    return job_types

def load_operationtypes():
    # 加载operationTypes.xlsx 文件
    df_operationtypes = pd.read_excel("dataset/example_operationtypes.xlsx")
    # 转换为字典：key = operationTypeId, value = (operationTypeName, machineTypeId, processingTime)
    operation_types = df_operationtypes.set_index("operationTypeId")[["operationTypeName", "machineTypeId", "processingTime"]].to_dict(orient="index")
    return operation_types


def load_problem():
    # 加载problem.xlsx文件
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
    (2, True,  True):  6.1,
    (2, True,  False): 6.2,
    (2, False, True):  6.3,
    (2, False, False): 6.4,
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
        print('初始化需求成功，当前产品名称，及其需求量为：',self.problem_info[0]['ProductionRequirements'])

        # 初始化操作类型列表，确保状态向量顺序一致
        self.operation_types_list = list(self.operation_types.keys())
        self.num_operation_types = len(self.operation_types_list)

        # 初始化机器状态
        self.init_machinestaus()
        print("初始化机器状态成功，当前机器状态为：", self.machine_status)

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
        """
        包括三个向量，第一个是等待操作的数量，第二个是闲置的机器数量，第三个是处理中的操作数量
        """

        # 向量一：等待操作的数量
        waiting_ops = np.zeros(self.num_operation_types)
        executable_operations = self.executable_operations
        # 把operation展开成一个列表，如果operation在可执行的operation中，就是需求量，否则是0
        state1 = []
        for job in self.operationrequirements:
            for operation in self.operationrequirements[job]:
                if operation in executable_operations:
                    state1.append(self.operationrequirements[job][operation])
                else:
                    state1.append(0)

        # 向量二：闲置的机器数量
        # 获得所有机器以及状态
        machinestatus = self.machine_status
        print("当前机器的状态：", machinestatus)
        state2 = []

        for job in self.operationrequirements:
            for operation in self.operationrequirements[job]:
                # 获取该操作的名称
                operation_name = self.operation_type_name[operation]
                # 统计可以处理此操作的空闲机器数量
                idle_machines = 0
                 # 读取所有机器的设置
                for machine in machinestatus:
                    machine_operation_setting = machinestatus[machine]['setting']
                    # 检查机器是否空闲且能处理该操作
                    if not machinestatus[machine]['working']:
                        # 检查机器是否可以处理该操作
                        # 假设设置格式为：job_operationName，例如 "A_DA1"
                        machine_job = machine_operation_setting.split('_')[0]
                        if machine_job == job:
                            # 进一步检查操作类型是否匹配
                            if operation_name in machine_operation_setting:
                                idle_machines += 1
                
                state2.append(idle_machines)

        # 向量三：处理中的操作数量
        state3 = []
        # 如果机器的状态为工作，就是1，否则是0
        for job in self.operationrequirements:
            for operation in self.operationrequirements[job]:
                # 获取该操作的名称
                operation_name = self.operation_type_name[operation]
                # 统计正在处理此操作的机器数量
                working_machines = 0
                for machine in machinestatus:
                    machine_operation_setting = machinestatus[machine]['setting']
                    # 检查机器是否正在处理该操作
                    if machinestatus[machine]['working']:
                        # 检查机器是否正在处理该操作
                        # 假设设置格式为：job_operationName，例如 "A_DA1"
                        machine_job = machine_operation_setting.split('_')[0]
                        if machine_job == job:
                            # 进一步检查操作类型是否匹配
                            if operation_name in machine_operation_setting:
                                working_machines += 1
                state3.append(working_machines)
        
        # 对三个向量进行归一化
        
        # 等待操作数量/总作业数量
        state1 = np.array(state1) / sum([num for job, num in self.jobrequirements])

        # 闲置机器数量/机器数量
        state2 = np.array(state2) / len(machinestatus)

        # 处理中的操作数量/机器数量
        state3 = np.array(state3) / len(machinestatus)

        print("当前状态为：", np.concatenate([state1, state2, state3]))

        return np.concatenate([state1, state2, state3])
    
    
    
    def execute_action(self, action):
        """
        
        获取原始动作，判断原始动作与可能执行的动作之间的距离，选择最短距离的可能动作进行执行
        （设置时间，加工时间，剩余操作的数量，剩余操作所需要的处理时间）

        """
        # 记录总时间

        # 获得可能执行的动作
        action_machine_pair = []

        # 首先获得可进行的操作
        executable_operations = self.executable_operations

        # 判断设置时间
        def setting_time(operation, machine):
            # 根据机器名称前缀确定机器类型ID
            if machine.startswith("DARES"):
                machine_type_id = 1
            elif machine.startswith("WBRES"):
                machine_type_id = 2
            else:
                raise ValueError(f"未知的机器类型: {machine}")
            # 当前机器的setting
            machine_setting = self.machine_status[machine]['setting']
            # 获取操作的名称
            operation_name = self.operation_type_name[operation]
            # 判断Jobtype是否相同
            is_job_type_same = machine_setting.startswith(operation_name.split('_')[0])
            # 判断OperationType是否相同，根据_后面的内容判断
            is_operation_type_same = operation_name.split('_')[1] in machine_setting
            # 获取设置时间
            setup_time = get_setup_time(machine_type_id, is_job_type_same, is_operation_type_same)

            return setup_time

        for operation in executable_operations:
            # 遍历所有空闲的机器
            for machine in self.machine_status:
                # 判断机器是否空闲
                if not self.machine_status[machine]['working']:

                    setup_time_value = setting_time(operation, machine)
                    progress_time = self.operation_types[operation]['processingTime']

                    # 剩余操作数量
                    job_type = next(job for job, ops in self.operationrequirements.items() if operation in ops)
                    total_operations = len(self.job_types[job_type])
                    current_operation_index = self.job_types[job_type].index(operation)
                    left_operations = total_operations - current_operation_index - 1

                    # 剩余操作所需要的总时间
                    left_time = 0
                    for i in range(current_operation_index + 1, total_operations):
                        left_time += self.operation_types[self.job_types[job_type][i]]['processingTime']
                    
                    action_machine_pair.append((setup_time_value, progress_time, left_operations, left_time))
        print("可能执行的动作为：", action_machine_pair)

        # 计算原始动作与可能执行动作之间的欧几里德距离
        def distance(action1, action2):
            return sum((a1 - a2) ** 2 for a1, a2 in zip(action1, action2)) ** 0.5
        
        # 选择距离最近的作为action
        action = min(action_machine_pair, key=lambda x: distance(action, x))
        print("选择的动作为：", action)

        # 执行动作，更新系统状态
        # 判断执行的是哪个operation
        def operation_to_execute(action):
            progress_time = action[1]
            left_operations = action[2]
            left_time = action[3]
            # 循环所有的operation找到对应的operation
            for operation in self.executable_operations:
                # 首先progerss时间要相同
                if progress_time == self.operation_types[operation]['processingTime']:
                    # 剩余操作数量要相同
                    job_type = next(job for job, ops in self.operationrequirements.items() if operation in ops)
                    total_operations = len(self.job_types[job_type])
                    current_operation_index = self.job_types[job_type].index(operation)
                    if left_operations == total_operations - current_operation_index - 1:
                        # 剩余操作所需要的总时间要相同
                        left_time_ = 0
                        for i in range(current_operation_index + 1, total_operations):
                            left_time_ += self.operation_types[self.job_types[job_type][i]]['processingTime']
                        if left_time == left_time_:
                            return operation
            return operation
        def machine_to_execute(action):
            setup_time = action[0]
            for machine in self.machine_status:
                if setup_time == setting_time(operation_to_execute(action), machine):
                    return machine
            return machine
        # 执行操作
        operation = operation_to_execute(action)
        machine = machine_to_execute(action)
        # 更新机器状态
        self.machine_status[machine]['working'] = True
        # 更新操作状态
        job_type = next(job for job, ops in self.operationrequirements.items() if operation in ops)
        self.operationrequirements[job_type][operation] -= 1
        # 更新可执行操作
        self.executable_operations = get_executable_operations(self.operationrequirements)
        print("执行动作后的机器状态：", self.machine_status)
        print("执行动作后的操作状态：", self.operationrequirements)
        print("执行动作后的可执行操作：", self.executable_operations)
        

        

    def calculate_reward(self):        
        pass
    
    def is_done(self):
        pass

    def step(self, action):
        # 执行动作
        self.execute_action(action)
        # 更新状态
        state = self.state()
        # 计算奖励
        reward = self.calculate_reward()
        # 判断是否结束
        done = self.is_done()
        return state, reward, done, {}
                    
    def reset(self):
        pass
    



# Example usage
if __name__ == "__main__":
    semenv = SemiconductorEnv()
    yuanaction = [6,1,1,1]
    semenv.execute_action(yuanaction)
    yuanaction1 = [6,1,1,1]
    semenv.execute_action(yuanaction1)