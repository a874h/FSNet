"""
mamba activate cvxpy
python make_dataset_AH.py 
"""

import cvxpy as cp
import numpy as np
import pickle
from scipy.sparse import lil_matrix, csr_matrix, block_diag

# --- System Parameters (Example Values - you can modify these) ---
N_B = 3  # Number of buses
N_G = 2  # Number of generators
N_L = 3  # Number of lines (not strictly needed for matrix construction, but for data definition)
T = 3    # Number of time periods

num_examples = 1000

seed = 2025
np.random.seed(seed)

# Generator Data:
# Each dictionary defines a generator with its properties.
# 'bus_idx': The bus where the generator is connected (0-indexed).
# 'P_min', 'P_max': Minimum and maximum active power output.
# 'cost_a', 'cost_b': Quadratic (a) and linear (b) cost coefficients for objective function (0.5*a*PG^2 + b*PG).
# 'ramp_up', 'ramp_down': Maximum positive/negative change in power output between periods.
generators_data = [
    {'bus_idx': 0, 'P_min': 0, 'P_max': 100, 'cost_a': 0.1, 'cost_b': 10, 'ramp_up': 50, 'ramp_down': 50},
    {'bus_idx': 1, 'P_min': 0, 'P_max': 120, 'cost_a': 0.05, 'cost_b': 12, 'ramp_up': 60, 'ramp_down': 60},
]

# Line Data:
# Each dictionary defines a transmission line.
# 'from_bus', 'to_bus': Connected buses (0-indexed).
# 'susceptance': The susceptance (B_ij) of the line, which determines power flow.
# 'flow_limit': Maximum power flow magnitude allowed on the line.
lines_data = [
    {'from_bus': 0, 'to_bus': 1, 'susceptance': 20, 'flow_limit': 100},
    {'from_bus': 0, 'to_bus': 2, 'susceptance': 15, 'flow_limit': 80},
    {'from_bus': 1, 'to_bus': 2, 'susceptance': 25, 'flow_limit': 110},
]

# Load Data:
# Each dictionary defines demand at a bus for each time period.
# 'bus_idx': The bus where the load is located (0-indexed).
# 'demand': A list of demand values for each time period [demand_t0, demand_t1, ...].
load_data = [
    {'bus_idx': 0, 'demand': [30, 40, 35]},
    {'bus_idx': 1, 'demand': [50, 45, 55]},
    {'bus_idx': 2, 'demand': [40, 55, 40]},
]

ref_bus_idx = 0 # Reference bus (its angle is fixed to 0 for all time periods)

# Angle limits (common for all buses and all time periods)
theta_min = -np.pi / 2 # -90 degrees
theta_max = np.pi / 2  # +90 degrees

# --- Derived Parameters ---
# Number of variables per time period: (Generators' power output + Buses' voltage angles)
num_vars_per_period = N_G + N_B
# Total number of optimization variables across all time periods
num_var = T * num_vars_per_period

# --- Initialize placeholder matrices/vectors using SciPy's LIL format for efficient construction ---
# (LIL is good for incremental building, then convert to CSR/CSC for computation)
Q = lil_matrix((num_var, num_var))
p = np.zeros(num_var)

L = np.zeros(num_var) # Lower bounds for y
U = np.zeros(num_var) # Upper bounds for y

# For A @ y == Xi (Equality Constraints)
A_rows_list = [] # List to store row indices for A
A_cols_list = [] # List to store column indices for A
A_data_list = [] # List to store non-zero values for A
#Xi_data_list = [] # List to store values for Xi (right-hand side of equalities)

# For G @ y <= h (Inequality Constraints)
G_rows_list = [] # List to store row indices for G
G_cols_list = [] # List to store column indices for G
G_data_list = [] # List to store non-zero values for G
h_data_list = [] # List to store values for h (right-hand side of inequalities)

# --- Helper functions for mapping (t, variable_idx) to global 'y' index ---
def get_pg_idx(t, gen_idx):
    """
    Returns the global index in the 'y' vector for generator 'gen_idx'
    (0-indexed) at time period 't' (0-indexed).
    """
    return t * num_vars_per_period + gen_idx

def get_theta_idx(t, bus_idx):
    """
    Returns the global index in the 'y' vector for bus 'bus_idx'
    (0-indexed) angle at time period 't' (0-indexed).
    """
    return t * num_vars_per_period + N_G + bus_idx

# --- Construct Q and p (Objective Function: 0.5 * y'Qy + p'y) ---
# Objective: Minimize sum over time of (a_k * PG_k^2 + b_k * PG_k)
for t in range(T):
    for k in range(N_G):
        gen_info = generators_data[k]
        pg_global_idx = get_pg_idx(t, k)
        
        # Quadratic part: 0.5 * (2 * cost_a) * PG^2 -> Q[idx,idx] = 2 * cost_a
        Q[pg_global_idx, pg_global_idx] = 2 * gen_info['cost_a']
        
        # Linear part: cost_b * PG -> p[idx] = cost_b
        p[pg_global_idx] = gen_info['cost_b']

# Convert Q to CSC format for efficient matrix-vector products with CVXPY
Q = Q.tocsc()

# --- Construct L and U (Box Constraints: L <= y <= U) ---
for t in range(T):
    # Generator output limits
    for k in range(N_G):
        gen_info = generators_data[k]
        pg_global_idx = get_pg_idx(t, k)
        L[pg_global_idx] = gen_info['P_min']
        U[pg_global_idx] = gen_info['P_max']
    
    # Bus voltage angle limits
    for b in range(N_B):
        theta_global_idx = get_theta_idx(t, b)
        L[theta_global_idx] = theta_min
        U[theta_global_idx] = theta_max

# --- Construct A and Xi (Equality Constraints: A @ y == Xi) ---
current_A_row = 0 # Keep track of the current row being added to A

# 1. Nodal Power Balance Equations: P_G_i - P_D_i - Sum_j (B_ij * (theta_i - theta_j)) = 0
# Rearranged: P_G_i - Sum_j (B_ij * theta_i) + Sum_j (B_ij * theta_j) = P_D_i
# (where Sum_j (B_ij) for a given i is effectively B_ii in the susceptance matrix B_bus)

# Create the Bus Susceptance Matrix (B_bus)
# B_bus[i,i] = sum of susceptances of lines connected to bus i
# B_bus[i,j] = -susceptance of line between i and j (if exists)
B_bus = np.zeros((N_B, N_B))
for line in lines_data:
    from_b, to_b, susceptance = line['from_bus'], line['to_bus'], line['susceptance']
    B_bus[from_b, from_b] += susceptance
    B_bus[to_b, to_b] += susceptance
    B_bus[from_b, to_b] -= susceptance
    B_bus[to_b, from_b] -= susceptance

for t in range(T):
    for b in range(N_B): # For each bus 'b'
        # Check if any generator is connected to this bus 'b'
        gen_at_bus = False
        for k, gen_info in enumerate(generators_data):
            if gen_info['bus_idx'] == b:
                # Add coefficient for generator active power output (PG)
                A_rows_list.append(current_A_row)
                A_cols_list.append(get_pg_idx(t, k))
                A_data_list.append(1.0) # Coefficient for PG
                gen_at_bus = True
        
        # Add coefficients for voltage angles (theta) using B_bus matrix
        for b_col in range(N_B): # Iterate through all possible connected buses
            if B_bus[b, b_col] != 0:
                A_rows_list.append(current_A_row)
                A_cols_list.append(get_theta_idx(t, b_col))
                A_data_list.append(-B_bus[b, b_col]) # Coefficient for theta_b_col

        # Right-hand side is the demand at this bus for this time period
        #Xi_data_list.append(load_data[b]['demand'][t])  # AH
        current_A_row += 1

# 2. Reference Bus Angle Constraint: theta_ref_bus_t = 0
for t in range(T):
    A_rows_list.append(current_A_row)
    A_cols_list.append(get_theta_idx(t, ref_bus_idx))
    A_data_list.append(1.0) # Coefficient for the reference bus angle
    #Xi_data_list.append(0.0) # Right-hand side is 0
    current_A_row += 1

# Convert lists to CSR matrix for efficient row-based operations (like slicing for constraints)
num_A_rows = current_A_row
A = csr_matrix((A_data_list, (A_rows_list, A_cols_list)), shape=(num_A_rows, num_var))
num_eq = A.shape[0]
#Xi = np.array(Xi_data_list) # Convert Xi to a NumPy array

# --- Construct G and h (Inequality Constraints: G @ y <= h) ---
current_G_row = 0 # Keep track of the current row being added to G

# 1. Line Flow Limits: -F_max <= B_ij * (theta_i - theta_j) <= F_max
# This translates to two inequalities for each line:
#   a) B_ij * (theta_i - theta_j) <= F_max
#   b) -B_ij * (theta_i - theta_j) <= F_max  (equivalent to B_ij * (theta_i - theta_j) >= -F_max)
for t in range(T):
    for line in lines_data:
        from_b, to_b, susceptance, flow_limit = line['from_bus'], line['to_bus'], line['susceptance'], line['flow_limit']
        
        # Inequality (a): susceptance * (theta_from - theta_to) <= flow_limit
        G_rows_list.append(current_G_row)
        G_cols_list.append(get_theta_idx(t, from_b))
        G_data_list.append(susceptance)
        G_rows_list.append(current_G_row)
        G_cols_list.append(get_theta_idx(t, to_b))
        G_data_list.append(-susceptance)
        h_data_list.append(flow_limit)
        current_G_row += 1
        
        # Inequality (b): -susceptance * (theta_from - theta_to) <= flow_limit
        G_rows_list.append(current_G_row)
        G_cols_list.append(get_theta_idx(t, from_b))
        G_data_list.append(-susceptance)
        G_rows_list.append(current_G_row)
        G_cols_list.append(get_theta_idx(t, to_b))
        G_data_list.append(susceptance)
        h_data_list.append(flow_limit)
        current_G_row += 1

# 2. Generator Ramping Limits:
#   PG_k_t - PG_k_{t-1} <= R_up_k
#   PG_k_{t-1} - PG_k_t <= R_down_k
# These constraints link variables across time periods.
for t in range(1, T): # Ramping applies from the second period onwards (t=1 refers to time period 2)
    for k in range(N_G):
        gen_info = generators_data[k]
        
        # Ramp Up constraint: P_G_t - P_G_{t-1} <= R_up_k
        G_rows_list.append(current_G_row)
        G_cols_list.append(get_pg_idx(t, k))   # P_G_t
        G_data_list.append(1.0)
        G_rows_list.append(current_G_row)
        G_cols_list.append(get_pg_idx(t-1, k)) # P_G_{t-1}
        G_data_list.append(-1.0)
        h_data_list.append(gen_info['ramp_up'])
        current_G_row += 1
        
        # Ramp Down constraint: P_G_{t-1} - P_G_t <= R_down_k
        G_rows_list.append(current_G_row)
        G_cols_list.append(get_pg_idx(t-1, k)) # P_G_{t-1}
        G_data_list.append(1.0)
        G_rows_list.append(current_G_row)
        G_cols_list.append(get_pg_idx(t, k))   # P_G_t
        G_data_list.append(-1.0)
        h_data_list.append(gen_info['ramp_down'])
        current_G_row += 1

# Convert lists to CSR matrix
num_G_rows = current_G_row
G = csr_matrix((G_data_list, (G_rows_list, G_cols_list)), shape=(num_G_rows, num_var))
h = np.array(h_data_list) # Convert h to a NumPy array
num_ineq = G.shape[0]
# --- Construct Xi (Equality Constraints: A @ y == Xi) ---  AH
# Construct Xi
X = np.zeros((num_examples,num_eq))
for i in range(num_examples):
    Xi_data_list = [] # List to store values for Xi (right-hand side of equalities)
    # get noise
    noise = np.random.randn()
    # loop
    for t in range(T):
        for b in range(N_B): # For each bus 'b'
            # Right-hand side is the demand at this bus for this time period
            Xi_data_list.append(load_data[b]['demand'][t] + noise)
    # 2. Reference Bus Angle Constraint: theta_ref_bus_t = 0
    for t in range(T):
        Xi_data_list.append(0.0) # Right-hand side is 0        
    # record
    X[i,:] = np.array(Xi_data_list)

# check size
print(num_ineq, num_eq)
"""
Q = np.diag(np.random.rand(num_var)*0.5)
p = np.random.uniform(-1, 1, num_var)
A = np.random.uniform(-1, 1, size=(num_eq, num_var))
X = np.random.uniform(-0.5, 0.5, size=(num_examples, num_eq))
G = np.random.uniform(-1, 1, size=(num_ineq, num_var))
h = np.sum(np.abs(G@np.linalg.pinv(A)), axis=1)
L = np.ones((num_var))*-5
U = np.ones((num_var))*5
"""
data = {'Q':Q.todense(),
        'p':p,
        'A':A.todense(),
        'X':X,
        'G':G.todense(),
        'h':h,
        'L':L,
        'U':U,
        'Y':[],
        #'XL':0.,     # AH MOD (else an error pops up). rescaling. used in training.py l.190
        #'XU':1.,      # AH MOD
        #'YL':0.,     # AH MOD (else an error pops up)
        #'YU':1.,      # AH MOD
        'XL': X.min(axis=0),     # https://github.com/MOSSLab-MIT/FSNet/blob/main/datasets/convex/qp/make_data.py
        'XU':X.max(axis=0),      # https://github.com/MOSSLab-MIT/FSNet/blob/main/datasets/convex/qp/make_data.py
        'YL':L,     # https://github.com/MOSSLab-MIT/FSNet/blob/main/datasets/convex/qp/make_data.py
        'YU':U,      #https://github.com/MOSSLab-MIT/FSNet/blob/main/datasets/convex/qp/make_data.py
        }
Y = []
for n in range(num_examples):
    Xi = X[n]
    y = cp.Variable(num_var)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, Q) + p.T @ y),
                        [G @ y <= h, y <= U, y >= L,
                        A @ y == Xi])
    prob.solve()
    print(n, np.max(y.value), np.min(y.value), y.value[0:5].T, end='\r')
    n+=1
    Y.append(y.value)
data['Y'] = np.array(Y)




i = 0
det_min = 0
best_partial = 0
while i < 1000:
    np.random.seed(i)
    partial_vars = np.random.choice(num_var, num_var - num_eq, replace=False)
    other_vars = np.setdiff1d(np.arange(num_var), partial_vars)
    _, det = np.linalg.slogdet(A.todense()[:, other_vars])
    if det>det_min:
        det_min = det
        best_partial = partial_vars
    i += 1
print('best_det', det_min)
data['best_partial'] = best_partial


#with open("datasets/qp/random_qp_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
with open("random{}_qp_dataset_var{}_ineq{}_eq{}_ex{}".format(seed,num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    pickle.dump(data, f)
