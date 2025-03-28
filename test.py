from gurobipy import Model, GRB
import pandas as pd

# 读取表格
# operation_types = pd.read_excel('dataset\\example_operationtypes.xlsx')
# job_types = pd.read_excel('dataset\\example_jobtypes.xlsx')
# setup_times = pd.read_excel('dataset\\example_setupTime.xlsx')
# production_requirements = pd.read_excel('dataset\\example_problem.xlsx')
# machine_types = pd.read_excel('dataset\\machineTypes.xlsx')


operation_types = pd.read_excel('dataset\\operationTypes.xlsx')
job_types = pd.read_excel('dataset\\jobTypes.xlsx')
setup_times = pd.read_excel('dataset\\setupTime.xlsx')
production_requirements = pd.read_excel('dataset\\problem.xlsx')
machine_types = pd.read_excel('dataset\\machineTypes.xlsx')

# 选择特定的数据集（例如 dataset_index=D1, problem_index=1）
req = production_requirements[(production_requirements['dataset_index'] == 'D1') & 
                              (production_requirements['problem_index'] == 1)].iloc[0]


# 解析生产需求
prod_req = req['ProductionRequirements']  # "A_2,B_1"
job_counts = {item.split('_')[0]: int(item.split('_')[1]) for item in prod_req.split(',')}
# 例如：{'A': 2, 'B': 1}

# 分配作业标识符
job_type = {}
job_id = 1
for jt_name, count in job_counts.items():
    jt_id = job_types[job_types['jobTypeName'] == jt_name]['jobTypeId'].iloc[0]
    for _ in range(count):
        job_type[job_id] = jt_id
        job_id += 1
# 结果：{1: 1, 2: 2, 3: 1}，其中 1 表示 A，2 表示 B

# 操作序列字典
operation_sequence = {row['jobTypeId']: [int(x) for x in row['operationTypeSequence'].split(',')] 
                      for _, row in job_types.iterrows()}
# 例如：{1: [1, 2], 2: [3, 4]}

# 生成操作列表
operations = []
for l in job_type.keys():
    num_ops = len(operation_sequence[job_type[l]])
    operations.extend([(l, j+1) for j in range(num_ops)])
# 结果：[(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]

# 解析机器设置
machine_settings = req['MachineSetting'].split(',')
machines = {}
eta = {}
for k, setting in enumerate(machine_settings, 1):
    machine_name, op_type_name = setting.split('_', 1)
    mt_name = machine_types[machine_types['machineTypeId'] == 
                            operation_types[operation_types['operationTypeName'] == op_type_name]
                            ['machineTypeId'].iloc[0]]['machineTypeName'].iloc[0]
    machines[k] = mt_name  # 例如：{1: 'DARES', 2: 'WBRES'}
    op_type_id = operation_types[operation_types['operationTypeName'] == op_type_name]['operationTypeId'].iloc[0]
    jt_id = job_types[job_types['operationTypeSequence'].str.contains(str(op_type_id))]['jobTypeId'].iloc[0]
    eta[k] = (jt_id, op_type_id)  # 例如：{1: (1, 1), 2: (2, 4)}
N_M = len(machines)  # 机器数量，例如 2

# 操作类型映射
operation_type = {(l, j): operation_sequence[job_type[l]][j-1] for l, j in operations}
# 例如：{(1,1): 1, (1,2): 2, (2,1): 3, (2,2): 4, (3,1): 1, (3,2): 2}

# 机器资格 E
E = {}
for l, j in operations:
    ot = operation_type[(l, j)]
    mt_id = operation_types[operation_types['operationTypeId'] == ot]['machineTypeId'].iloc[0]
    mt_name = machine_types[machine_types['machineTypeId'] == mt_id]['machineTypeName'].iloc[0]
    E[(l, j)] = [k for k, mt in machines.items() if mt == mt_name]
# 例如：{(1,1): [1], (1,2): [2], (2,1): [1], (2,2): [2], (3,1): [1], (3,2): [2]}

# 加工时间
p = {(l, j): operation_types[operation_types['operationTypeId'] == operation_type[(l, j)]]
     ['processingTime'].iloc[0] for l, j in operations}
# 例如：{(1,1): 2, (1,2): 3, (2,1): 1, (2,2): 4, (3,1): 2, (3,2): 3}

# 构建切换时间查找表
setup_time_dict = {(row['MachineTypeId'], row['isJobTypeSame'], row['isOperationTypeSame']): row['SetupTime'] 
                   for _, row in setup_times.iterrows()}

def setup_time(k, op1, op2):
    mt = machines[k]  # 机器类型名称
    mt_id = machine_types[machine_types['machineTypeName'] == mt]['machineTypeId'].iloc[0]
    # 如果 op1 是 eta，格式为 (job_type, operation_type)
    if isinstance(op1, tuple) and len(op1) == 2:
        i1, ot1 = op1
    else:
        i1, ot1 = job_type[op1[0]], operation_type[op1]
    i2, ot2 = job_type[op2[0]], operation_type[op2]
    is_job_type_same = (i1 == i2)
    is_operation_type_same = (ot1 == ot2)
    return setup_time_dict.get((mt_id, is_job_type_same, is_operation_type_same), 100)

# 总操作数
o_max = len(operations)  # 例如 6

# 大 M 参数
H = sum(p.values()) + max(setup_times['SetupTime']) * o_max  # 保守估计


# Create model
# 创建模型
model = Model("Semiconductor_Scheduling")

# 变量
x = model.addVars([(h,k,l,j) for h in range(1,o_max+1) for k in range(1,N_M+1) 
                   for l,j in operations], vtype=GRB.BINARY, name="x")
z = model.addVars([(h,k) for h in range(1,o_max+1) for k in range(1,N_M+1)], 
                  vtype=GRB.BINARY, name="z")
c_bar = model.addVars([(h,k) for h in range(1,o_max+1) for k in range(1,N_M+1)], 
                      vtype=GRB.CONTINUOUS, lb=0, name="c_bar")
c = model.addVars(operations, vtype=GRB.CONTINUOUS, lb=0, name="c")
C_max = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="C_max")

# 目标
model.setObjective(C_max, GRB.MINIMIZE)

# Constraints
# 1. Operation assignment
for l,j in operations:
    # i = job_type[l]
    model.addConstr(sum(x[h,k,l,j] for h in range(1,o_max+1) for k in E[l,j]) == 1)

# 2. Position occupancy
for h in range(1,o_max+1):
    for k in range(1,N_M+1):
        model.addConstr(sum(x[h,k,l,j] for l,j in operations) == z[h,k])

# 3. Position continuity
for h in range(1,o_max):
    for k in range(1,N_M+1):
        model.addConstr(z[h+1,k] <= z[h,k])

# 4. Machine eligibility
for l,j in operations:
    # i = job_type[l]
    for h in range(1,o_max+1):
        for k in range(1,N_M+1):
            if k not in E[l,j]:
                model.addConstr(x[h,k,l,j] == 0)

# 5. Completion time linking (strengthened)
for l,j in operations:
    for h in range(1,o_max+1):
        for k in range(1,N_M+1):
            model.addConstr(c[l,j] >= c_bar[h,k] - H * (1 - x[h,k,l,j]))
            model.addConstr(c[l,j] <= c_bar[h,k] + H * (1 - x[h,k,l,j]))

# 6. Makespan
for l,j in operations:
    model.addConstr(C_max >= c[l,j])

# 7. Job-internal sequence (corrected)
for l in job_type.keys():
    # i = job_type[l]
    if (l, 1) in operations and (l, 2) in operations:
        for h in range(1, o_max + 1):
            for k in range(1, N_M + 1):
                model.addConstr(c_bar[h,k] - p[l, 2] >= c[l, 1] - H * (1 - x[h,k,l,2]))

# 8a. 第一个操作的切换时间
for k in range(1, N_M+1):
    for l, j in operations:
        sigma = setup_time(k, eta[k], (l, j))
        model.addConstr(c_bar[1,k] >= p[l,j] + sigma - H * (1 - x[1,k,l,j]))

# 8b. 后续操作的切换时间
for h in range(1, o_max):
    for k in range(1, N_M+1):
        for l, j in operations:
            for l_prime, j_prime in operations:
                if (l, j) != (l_prime, j_prime):
                    sigma = setup_time(k, (l_prime, j_prime), (l, j))
                    model.addConstr(c_bar[h+1, k] >= c_bar[h, k] + p[l, j] + sigma - 
                                   H * (2 - x[h, k, l_prime, j_prime] - x[h+1, k, l, j]))

# 优化
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"Optimal Makespan: {C_max.x}")
    for k in range(1, N_M+1):
        print(f"Machine {k} Schedule:")
        for h in range(1, o_max+1):
            if z[h,k].x > 0.5:
                for l, j in operations:
                    if x[h,k,l,j].x > 0.5:
                        print(f"  Position {h}: Operation o_{l},{j}, Completion Time {c_bar[h,k].x}")
else:
    print("No optimal solution found")