from torch.utils.data import DataLoader
import pickle 
from utils.optimization_utils import *

val_size=256
test_size=256
seed=2025
filepath = 'dataset/convex/qp/random2025_qp_dataset_var192_ineq238_eq120_ex1000'
print('filepath=',filepath)
with open(filepath, 'rb') as f:
    dataset = pickle.load(f)
# Create problem instance using the appropriate class
data = QPProblem(dataset, val_size, test_size, seed)
