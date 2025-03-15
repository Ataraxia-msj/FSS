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
    file_path = "dataset/operationTypes.xlsx"
    df = pd.read_excel(file_path)

    # 假设 ID 在第一列，名称在第二列（请根据实际列名调整）
    id_column = df.columns[0]  # 第一列作为 ID
    name_column = df.columns[1]  # 第二列作为名称
    id_name_dict = dict(zip(df[id_column], df[name_column]))

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


        


class SemiconductorEnv:
    def __init__(self):

        
        # 初始化需求
        # 加载数据
        self.job_types = load_jobtypes()
        self.operation_types = load_operationtypes()
        self.machine_types = {1: 'DARES', 2: 'WBRES'}
        self.problem_info = load_problem()
        self.machine_status = {}
        self.operation_type_name = get_operation_type_name()
        print('初始化需求成功，当前产品名称，及其需求量为：',self.problem_info[0]['ProductionRequirements'])


        # 初始化机器状态
        self.init_machinestaus()
        print("初始化机器状态成功，当前机器状态为：", self.machine_status)
        
    def init_machinestaus(self):
        machinesetting = self.problem_info[0]['MachineSetting']
        for machine, setting in machinesetting:
            self.machine_status[machine] = {
                'setting': setting,
                'working': False  # 默认为不工作状态
            }
        
    def state(self):
        # 包括三个向量，第一个是等待操作的数量，第二个是闲置的机器数量，第三个是处理中的操作数量

        # 向量一：等待操作的数量

        #获得所有job，和对应的operation
        jobtypes = self.job_types   #{'A': [1, 2], 'B': [3, 4]}
        # 每个job的需求量
        problems = self.problem_info
        jobrequirements = problems[0]['ProductionRequirements']  # [('A', 2), ('B', 1)]
        # 获得每个operation的需求量,{A:{1:2, 2:2},B:{3:1, 4:1}}
        operationrequirements = {}
        for job, num in jobrequirements:
            for operation in jobtypes[job]:
                if job not in operationrequirements:
                    operationrequirements[job] = {}
                operationrequirements[job][operation] = num
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
        # 获得当前可执行的operation
        self.executable_operations = get_executable_operations(operationrequirements)
        executable_operations = self.executable_operations
        # 把operation展开成一个列表，如果operation在可执行的operation中，就是需求量，否则是0
        state1 = []
        for job in operationrequirements:
            for operation in operationrequirements[job]:
                if operation in executable_operations:
                    state1.append(operationrequirements[job][operation])
                else:
                    state1.append(0)


        # 向量二：闲置的机器数量

        # 获得所有机器以及状态
        machinestatus = self.machine_status
        print("当前机器的状态：", machinestatus)
        state2 = []

        for job in operationrequirements:
            for operation in operationrequirements[job]:
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
        for job in operationrequirements:
            for operation in operationrequirements[job]:
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
        state1 = np.array(state1) / sum([num for job, num in jobrequirements])

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

        # 获得可能执行的动作

        action_machine_pair = []

        # 首先获得可进行的操作
        executable_operations = self.executable_operations

        print("可执行的操作为：", executable_operations)





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
    semenv.execute_action(1)