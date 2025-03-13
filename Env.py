import numpy as np
import pandas as pd

# 处理jobtypes.xlsx文件
def load_jobtypes(): 
    # 加载 jobTypes.xlsx 文件
    df_jobtypes = pd.read_excel("FSSP/dataset/example_jobtypes.xlsx")
    # 解析 operationTypeSequence 字段，将字符串转换为整数列表
    df_jobtypes["operationTypeSequence"] = df_jobtypes["operationTypeSequence"].apply(lambda x: list(map(int, str(x).split(','))))
    # 转换为字典：key = jobTypeId, value = operationTypeSequence
    job_types = df_jobtypes.set_index("jobTypeName")["operationTypeSequence"].to_dict()
    return job_types

def load_operationtypes():
    # 加载operationTypes.xlsx 文件
    df_operationtypes = pd.read_excel("FSSP/dataset/example_operationtypes.xlsx")
    # 转换为字典：key = operationTypeId, value = (operationTypeName, machineTypeId, processingTime)
    operation_types = df_operationtypes.set_index("operationTypeId")[["operationTypeName", "machineTypeId", "processingTime"]].to_dict(orient="index")
    return operation_types


def load_problem():
    # 加载problem.xlsx文件
    df_problem = pd.read_excel("FSSP/dataset/example_problem.xlsx")
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

        # 加载数据
        self.job_types = load_jobtypes()
        self.operation_types = load_operationtypes()
        self.machine_types = {1: 'DARES', 2: 'WBRES'}
        self.problem_info = load_problem()
        self.machine_status = {}
        self.init_machinestaus()
        
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
        executable_operations = get_executable_operations(operationrequirements)
        
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
        

        

    def step(self, action):
        pass


    def reset(self):
        pass
    


# Example usage
if __name__ == "__main__":
    semenv = SemiconductorEnv()
    semenv.state()