from gurobipy import Model, GRB

# Create model
model = Model("Semiconductor_Scheduling")

# Parameters
N_M = 2
o_max = 6
H = 1000

operations = [(1,1), (1,2), (3,1), (3,2), (2,1), (2,2)]
job_type = {1: 1, 2: 2, 3: 1}
op_type = {(l,j): (job_type[l], j) for l,j in operations}
E = {(1,1): [1], (1,2): [2], (3,1): [1], (3,2): [2], (2,1): [1], (2,2): [2]}
p = {(1,1): 2, (1,2): 3, (3,1): 2, (3,2): 3, (2,1): 1, (2,2): 4}

def setup_time(op1, op2, E=E):
    k1 = E[op1[0], op1[1]][0]
    k2 = E[op2[0], op2[1]][0]
    is_job_type_same = op1[0] == op2[0]
    is_operation_type_same = op1[1] == op2[1]
    if k1 == 1 and k2 == 1:
        if is_job_type_same and is_operation_type_same:
            return 0
        elif is_job_type_same:
            return 3
        else:
            return 6
    elif k1 == 2 and k2 == 2:
        if is_job_type_same and is_operation_type_same:
            return 0
        elif is_job_type_same:
            return 6.2
        elif is_operation_type_same:
            return 6.3
        else:
            return 6.4
    else:
        return 100
        

eta = {1: (1,1), 2: (2,2)}

# Variables
x = model.addVars([(h,k,l,j) for h in range(1,o_max+1) for k in range(1,N_M+1) 
                   for l,j in operations], vtype=GRB.BINARY, name="x")
z = model.addVars([(h,k) for h in range(1,o_max+1) for k in range(1,N_M+1)], 
                  vtype=GRB.BINARY, name="z")
c_bar = model.addVars([(h,k) for h in range(1,o_max+1) for k in range(1,N_M+1)], 
                      vtype=GRB.CONTINUOUS, lb=0, name="c_bar")
c = model.addVars(operations, vtype=GRB.CONTINUOUS, lb=0, name="c")
C_max = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="C_max")

# Objective
model.setObjective(C_max, GRB.MINIMIZE)

# Constraints
# 1. Operation assignment
for l,j in operations:
    i = job_type[l]
    model.addConstr(sum(x[h,k,l,j] for h in range(1,o_max+1) for k in E[i,j]) == 1)

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
    i = job_type[l]
    for h in range(1,o_max+1):
        for k in range(1,N_M+1):
            if k not in E[i,j]:
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
    i = job_type[l]
    if (l, 1) in operations and (l, 2) in operations:
        for h in range(1, o_max + 1):
            for k in range(1, N_M + 1):
                model.addConstr(c_bar[h,k] - p[i, 2] >= c[l, 1] - H * (1 - x[h,k,l,2]))

# 8. Machine sequence
# a. First operation
for k in range(1,N_M+1):
    for l,j in operations:
        i = job_type[l]
        sigma = setup_time(eta[k], (i,j))
        model.addConstr(c_bar[1,k] >= p[i,j] + sigma - H * (1 - x[1,k,l,j]))

# b. Subsequent operations
for h in range(1, o_max):
    for k in range(1, N_M+1):
        for l, j in operations:
            for l_prime, j_prime in operations:
                if (l, j) != (l_prime, j_prime):
                    i = job_type[l]
                    i_prime = job_type[l_prime]
                    sigma = setup_time((i_prime, j_prime), (i, j))
                    model.addConstr(c_bar[h+1, k] >= c_bar[h, k] + p[i, j] + sigma - 
                                   H * (2 - x[h, k, l_prime, j_prime] - x[h+1, k, l, j]))

# Optimize
model.optimize()

# Output
if model.status == GRB.OPTIMAL:
    print(f"Optimal Makespan: {C_max.x}")
    for k in range(1,N_M+1):
        print(f"Machine {k} Schedule:")
        for h in range(1,o_max+1):
            if z[h,k].x > 0.5:
                for l,j in operations:
                    if x[h,k,l,j].x > 0.5:
                        print(f"  Position {h}: Operation o_{l},{j}, Completion Time {c_bar[h,k].x}")
else:
    print("No optimal solution found")