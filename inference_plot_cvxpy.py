"""
Run + plot inference on pre-trained FSNet NN
when dataset is built from cvxpy model in make_dataset_certes_cvxpy_v0.py

TODO:  load  config from: model_save_content['config']

Setup
----------

* Create a virtual environment with pytorch and cvxpy.
* Create a training dataset (e.g. `certes_lissi/LP_NN/FS_Net/make_dataset_certes_cvxpy_v0.py`).
* Train the model (see `certes_lissi/LP_NN/FS_Net/README_AH.py`).
* Copy `inference_plot_cvxpy.py` into `$HOME_FSNET/`

Run
-----------

```
mamba activate cvxpy 
# example with FSNet + `certes_lissi/LP_NN/FS_Net/make_dataset_certes_cvxpy_v0.py`
python inference_plot_cvxpy.py --method FSNet --prob_type convex --prob_name qp --seed 2025  --prob_size 192 238 120 1000 --batch_size 100 --test_size 100

# example with DC3 + `certes_lissi/LP_NN/FS_Net/make_dataset_certes_cvxpy_v0.py`
python inference_plot_cvxpy.py --method DC3 --prob_type convex --prob_name qp --seed 2025  --prob_size 192 238 120 1000 --batch_size 100 --test_size 100

#  penalty+skm
python inference_plot_cvxpy.py --method penalty --prob_type convex --prob_name qp --seed 2025  --prob_size 192 238 120 1000 --batch_size 100 --test_size 100
```
"""

# AH: new stuff
import sys
sys.path.append("/data/aurelien/local/git/certes_lissi/LP_NN")
from test_cvxpy_3_bus_24hrs_battery_PV import * # AH: this is new

# copied from test_cvxpy_3_bus_24hrs_battery_PV
# TODO: add as a CLI param ?
N_B = 3  # Number of buses
N_G = 2  # Number of generators
N_BATT = 1 # Number of battery storage units
T = 24    # Number of time periods (e.g., hours)
num_vars_per_period = N_G + N_B + N_BATT + N_BATT + N_BATT

import yaml
import torch
import time
import os, argparse
from utils.trainer import load_instance, create_model, Evaluator #, Trainer

# Define available problem types and problems
PROBLEM_TYPES = ['convex', 'nonconvex', 'nonsmooth_nonconvex']
PROBLEM_NAMES = ['qp', 'qcqp', 'socp']

def create_parser():
    """Create and configure the argument parser, then load and process the configuration."""
    parser = argparse.ArgumentParser(description='Neural Network Optimization')
    
    # General parameters
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--method', type=str, 
                        help='Training method (penalty, adaptive_penalty, FSNet, DC3, projection)')
    parser.add_argument('--prob_type', type=str, choices=PROBLEM_TYPES,
                        help='Problem type (convex, nonconvex, nonsmooth_nonconvex)')
    parser.add_argument('--prob_name', type=str, choices=PROBLEM_NAMES,
                        help='Problem name (qp, qcqp, socp)')
    parser.add_argument('--prob_size', type=int, nargs='+', default=[100, 50, 50, 10000],
                        help='Problem size parameters [n, m, p, N] (default: [100, 50, 50, 10000])')
    parser.add_argument('--network', type=str, default='MLP',
                        help='Type of neural network to use')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--ablation', type=bool, default=False)

    # Dataset parameters
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--val_size', type=int, help='Size of validation dataset')
    parser.add_argument('--test_size', type=int, help='Size of test dataset')
    parser.add_argument('--dropout', type=float, help='Dropout rate for the model')

    # Neural network parameters
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, help='Learning rate decay factor')
    parser.add_argument('--lr_decay_step', type=int, help='Learning rate decay step size')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, help='Number of hidden layers')
    
    # Feasibility seeking parameters
    parser.add_argument('--scale', type=float, help='Scale')
    parser.add_argument('--dist_weight', type=float, help='Distance weight')
    parser.add_argument('--max_diff_iter', type=int, help='Maximum number of iterations for keeping the track of gradient')

    args = parser.parse_args()
    
    # Load configuration from YAML file
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments if provided
    if args.method:
        config['seed'] = args.seed
    if args.method:
        config['method'] = args.method
    if args.prob_type:
        config['prob_type'] = args.prob_type
    if args.prob_name:
        config['prob_name'] = args.prob_name
    if args.prob_size:
        config['prob_size'] = args.prob_size
    if args.network:
        config['network'] = args.network
    
    # Override dataset parameters
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.val_size:
        config['val_size'] = args.val_size
    if args.test_size:
        config['test_size'] = args.test_size
    
    # Override neural network parameters
    if args.lr:
        config['lr'] = args.lr
    if args.lr_decay:
        config['lr_decay'] = args.lr_decay
    if args.lr_decay_step:
        config['lr_decay_step'] = args.lr_decay_step
    if args.num_epochs:
        config['num_epochs'] = args.num_epochs
    if args.hidden_dim:
        config['hidden_dim'] = args.hidden_dim
    if args.num_layers:
        config['num_layers'] = args.num_layers
    if args.dropout:
        config['dropout'] = args.dropout
    
    # Feasibility seeking parameters
    if args.scale:
        config['FSNet']['scale'] = args.scale
    if args.dist_weight is not None:
        config['FSNet']['dist_weight'] = args.dist_weight
    if args.max_diff_iter is not None:
        config['FSNet']['max_diff_iter'] = args.max_diff_iter

    # Ablation study flag
    config['ablation'] = args.ablation

    return args, config

def main():
    # Parse command-line arguments and get processed config
    args, config = create_parser()
    # TODO!!!!!!!!!! load config from: model_save_content['config']


    # Get the problem type and name from config (with defaults)
    prob_type = config.get('prob_type', 'Error')
    prob_name = config.get('prob_name', 'Error')
    
    print(f"\n======= Running for problem: {prob_type}/{prob_name} =======\n")

    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # Load data 
    print(f"Loading problem instance: {prob_type}/{prob_name} with size {config['prob_size']}")
    data, result_save_dir = load_instance(config)
    # Initialize model
    model = create_model(data, config['method'], config)

    # load model state (copied from trainer.py: _save_model_and_results)
    """NB AH: you can try commenting below, to see the impact on metrics (it degrades badly)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_filename = f"model_seed{config.get('seed', 'N_A')}.pt"
    model_filepath = os.path.join(result_save_dir, model_filename)
    print('load model state_dict from: ', model_filepath)
    model_save_content= torch.load(model_filepath, map_location=device,weights_only=False)
    model.load_state_dict(model_save_content['model_state_dict'])

    # Initialize Evaluator    
    evaluator = Evaluator(data, config['method'], config)
                
    # copied from trainer.py: Trainer.train()
    if hasattr(data, 'test_dataset'):
        # Get test batch sizes from config or use defaults
        test_batch_sizes = config.get('test_batch_sizes', [256, 512]) 
        print(f"Testing with batch sizes: {test_batch_sizes}")
        # Run evaluation with all batch sizes and collect detailed results for all
        batch_size_results, all_detailed_results = evaluator.evaluate_multiple_batch_sizes(
            model, 
            data.test_dataset, 
            test_batch_sizes, 
            "test"
        )
        print('all_detailed_results', all_detailed_results )
    else:
        raise ValueError('data.test_dataset does not exist')
    return all_detailed_results

def plot(results):
    """
    copied from: certes_lissi/LP_NN$ mousepad test_cvxpy_3_bus_24hrs_battery_PV.py
    AH func
     Q !!!! why results[256] is working ??? 

    parameters:
    ------------
    results: dict
    detailed results outputted by FSNet inference
    """
    # extract y predicted by nn
    # NB: size is 15 with make_dataset_AH.py
    Y_final = results[256][0]['Y_final']  # Q !!!! why results[256] is working ???  calling param is 100
    Y_true = results[256][0]['Y_true']  # Q !!!! why results[256] is working ???  calling param is 100
    # get just 1 sample 
    idx_sample = 20
    y_final = Y_final[idx_sample,:] 
    y_true = Y_true[idx_sample,:] 

    fig,ax = plt.subplots(2,2,figsize=(15,15))
    ax[0,0].set_xlabel('t')
    for k in range(N_G):
        pg_k=[]
        pg_true_k=[]
        for t in range(T):        
            #pg_global_idx = get_pg_idx(t, k)
            pg_global_idx =get_pg_idx(t, k,num_vars_per_period)
            pg_k.append(  y_final[pg_global_idx] )    
            pg_true_k.append(  y_true[pg_global_idx] )    
        print(f"pg[{k}]=",pg_k)
        ax[0,0].plot(range(T),pg_k, label='Gen '+ str(k)+' NN')
        ax[0,0].plot(range(T),pg_true_k,'--', label='Gen '+str(k)+' CVX' )
        ax[0,0].legend()
    ax[0,0].set_title('Load, generation power')

    # THETA
    for k in range(N_B):
        theta_k=[]
        theta_true_k=[]
        for t in range(T):        
            #pg_global_idx = get_pg_idx(t, k)
            theta_idx = get_theta_idx(t, k,num_vars_per_period,N_G)
            theta_k.append(  y_final[theta_idx] )    
            theta_true_k.append(  y_true[theta_idx] )    
        print(f"theta[{k}]=",pg_k)
        ax[0,1].plot(range(T),theta_k, label='bus '+str(k)+ ' NN')
        ax[0,1].plot(range(T),theta_true_k,"--", label='bus '+str(k)+ ' CVX')
        ax[0,1].legend()
    ax[0,1].set_title(r'$\theta_k$')
    ax[0,1].set_xlabel('t')

    # Battery
    batt_idx=0
    p_charge_k=[]
    p_charge_true_k=[]
    for t in range(T):        
        #p_charge_idx = get_p_charge_idx(t, batt_idx)
        p_charge_idx =get_p_charge_idx(t, batt_idx,num_vars_per_period,N_G, N_B)
        p_charge_k.append(  y_final[p_charge_idx] )    
        p_charge_true_k.append(  y_true[p_charge_idx] )    
    print(f"p_charge_k[{batt_idx}]=",p_charge_k)
    ax[1,0].plot(range(T),p_charge_k, label='Charge NN') 
    ax[1,0].plot(range(T),p_charge_true_k,'--', label='Charge CVX') 
    ax[1,0].set_xlabel('t')

    batt_idx=0
    p_discharge_k=[]
    p_discharge_true_k=[]
    for t in range(T):        
        #p_discharge_idx = get_p_discharge_idx(t, batt_idx)
        p_discharge_idx =get_p_discharge_idx(t, batt_idx,num_vars_per_period,N_G,N_B,N_BATT )
        p_discharge_k.append(  y_final[p_discharge_idx] )    
        p_discharge_true_k.append(  y_true[p_discharge_idx] )    
    print(f"p_discharge_k[{batt_idx}]=",p_discharge_k)
    ax[1,0].plot(range(T),p_discharge_k, label='Discharge NN') 
    ax[1,0].plot(range(T),p_discharge_true_k,'--', label='Discharge CVX') 
    ax[1,0].legend()
    ax[1,0].set_title('Battery charge/discharge')

    batt_idx=0
    e_batt_k=[]
    e_batt_true_k=[]
    for t in range(T):        
        #e_batt_k_idx = get_e_batt_idx(t, batt_idx)
        e_batt_k_idx =get_e_batt_idx(t, batt_idx,num_vars_per_period,N_G,N_B,N_BATT)
        e_batt_k.append(  y_final[e_batt_k_idx] )    
        e_batt_true_k.append(  y_true[e_batt_k_idx] )    
    print(f"e_batt_k[{batt_idx}]=",e_batt_k)
    ax[1,1].plot(range(T),e_batt_k, label='SOC NN') 
    ax[1,1].plot(range(T),e_batt_true_k,"--", label='SOC CVX') 
    ax[1,1].set_title('SOC')
    ax[1,1].legend()
    ax[1,1].set_xlabel('t')

    # plot violations
    # see HOME_FSNET/utils/optimization_utils.py: QPProblem(BaseProblem)
    #     eq_resid(), ineq_resid           

    #TODO ax[0,0].plot(range(T),d['demand'],"r--", label='demand')
    #TODO ax[0,0].plot(range(T),d['pv_generation'],"y--", label='pv') 
    plt.show()


if __name__ == "__main__":
    results = main()
    # load cvxpy model used to build dataset
    # read and plot results
    plot(results) # new function