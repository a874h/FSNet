"""
https://github.com/olavolav/uniplot
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uniplot import plot


def load_results(file_path):
    """
    Load results from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        dict: Dictionary containing the loaded results.
    """
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None 


#method = "penalty"#"DC3"##"FSNet" #
prob_type = "convex"
prob_name = "qp"
seed = 2025
"""
#fname = "Problem-100-50-50-10000"
#fname = "Problem-360-432-288-10000"
fname = "Problem-192-238-120-10000"

dir_path = f"results/{prob_type}/{prob_name}/{prob_name.upper()}{fname}/MLP_{method}/results_seed{seed}.pkl"
# --------------
# converge
df = pd.DataFrame(results['train_history'])
plot(df.loss.values)
"""
# ------------------
row_l = []
seed = 2025
#algo_l = ['DC3','FSNet','skm']
algo_l = ['penalty','FSNet','DC3','skm']
#algo_l = ['penalty','DC3','skm']
#algo_l = ['penalty','skm']
#algo_l = ['FSNet']
batch_size = 256
#fname = "Problem-100-50-50-10000"
fname = "Problem-192-238-120-1000"
#fname ="Problem-360-432-288-10000"
for method in algo_l:
    #dir_path = f"{prob_type}/{prob_name}/{prob_name.upper()}Problem-100-50-50-10000/MLP_{method}/results_seed{seed}.pkl"
    #dir_path = f"{prob_type}/{prob_name}/{prob_name.upper()}Problem-192-238-120-1000/MLP_{method}/results_seed{seed}.pkl"
    dir_path = f"results/{prob_type}/{prob_name}/{prob_name.upper()}{fname}/MLP_{method}/results_seed{seed}.pkl"
    print(dir_path)
    results = load_results(dir_path)
    # convergence 
    df = pd.DataFrame(results['train_history'])
    plot(df.loss.values, title=method)
    # Get batch comparison data from the loaded results
    batch_comparison = results['test_results']['batch_size_comparison']
    # get only batchsize 256
    metrics = (results['test_results']['batch_size_comparison'].pop(batch_size))['metrics']
    # AH !!!!!!
    avg_sample_time = metrics['avg_inference_time']/batch_size
    row = {
        'Algorithm': method,
            'Opt Gap Mean (\%)': f"{metrics['opt_gap_mean']*100:.4f}",
            #'Opt Gap Std (\%)': f"{metrics['opt_gap_std']*100:.4f}",
            #'Opt Gap Std (\%)': f"{metrics['opt_gap_std']*100:.2e}",
            'Opt Gap Max (\%)': f"{metrics['opt_gap_max']*100:.4f}",
            'Eq Viol Mean': f"{metrics['eq_violation_l1_mean']:.2e}",
            'Eq Viol Max': f"{metrics['eq_violation_l1_max']:.2e}",
            'Ineq Viol Mean': f"{metrics['ineq_violation_l1_mean']:.2e}",
            'Ineq Viol Max': f"{metrics['ineq_violation_l1_max']:.2e}",
            'Sample inference time (s)': f"{avg_sample_time:.2e}",
            'Train time(s)':   f"{results['training_time_seconds']:.2e}"
        
    }
    row_l.append(row)

df = pd.DataFrame(row_l)
#print(df)
print(df.T.to_latex())
