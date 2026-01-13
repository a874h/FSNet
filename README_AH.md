## FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees

AH stuff
-----------




scale: optimization_utils.py:scale_full (l.78)  
       Y.shape =(512,100)
       upper_bound.shape=(1,100)

python main.py --method FSNet --prob_type convex --prob_name qp

python main.py --method FSNet --prob_type convex --prob_name qp --seed 2025 --prob_size 15 26 12 1000 --batch_size 100 --val_size 100 --test size 100 --num_epochs 150

BUG
(SOLVED by commit: utils/trainer.py: l.329  #self.method_params['val_tol'] = np.clip( #MOD AH)


AH RESULTS
---------------

Epoch 150/150, Loss: 6104.3802, Obj: 6090.2345, Eq Viol (l1): 0.000004, Ineq Viol (l1): 0.000000, Epoch time: 4.65s

Running validation at epoch 150...

VALIDATION_EPOCH_150 EVALUATION RESULTS:

Objective Value:     6.112374e+03
True Objective:      6.112327e+03
Optimality Gap:      7.535509e-06 ± 0.000000e+00
Eq Violation l1:   1.581128e-06 (max: 1.200913e-04)
Ineq Violation l1: 0.000000e+00 (max: 0.000000e+00)
Solution Distance:   6.335924e-01 ± 0.000000e+00
Avg Inference Time:  0.1285s

Training completed in 796.28 seconds.


COMPREHENSIVE TEST EVALUATION WITH DETAILED RESULTS

Testing with batch sizes: [256, 512]

Evaluating with batch size: 256 (with detailed results)

TEST_BS256 EVALUATION RESULTS:
Objective Value:     6.084415e+03
True Objective:      6.084370e+03
Optimality Gap:      7.189490e-06 ± 0.000000e+00
Eq Violation l1:   7.719348e-06 (max: 7.276027e-04)
Ineq Violation l1: 0.000000e+00 (max: 0.000000e+00)
Solution Distance:   5.989099e-01 ± 0.000000e+00
Avg Inference Time:  0.1444s


Evaluating with batch size: 512 (with detailed results)

TEST_BS512 EVALUATION RESULTS:
Objective Value:     6.084415e+03
True Objective:      6.084370e+03
Optimality Gap:      7.189490e-06 ± 0.000000e+00
Eq Violation l1:   7.719348e-06 (max: 7.276027e-04)
Ineq Violation l1: 0.000000e+00 (max: 0.000000e+00)
Solution Distance:   5.989099e-01 ± 0.000000e+00
Avg Inference Time:  0.1427s


TEST BATCH SIZE COMPARISON:

Batch Size   Objective    Opt Gap      Eq Viol      Ineq Viol    Time (s)  
256          6.0844e+03   7.1895e-06   7.7193e-06   0.0000e+00   0.14      
512          6.0844e+03   7.1895e-06   7.7193e-06   0.0000e+00   0.14      


Saving model and results to: results/convex/qp/QPProblem-15-26-12-1000/MLP_FSNet
✓ Model saved: results/convex/qp/QPProblem-15-26-12-1000/MLP_FSNet/model_seed2025.pt
✓ Detailed results saved: results/convex/qp/QPProblem-15-26-12-1000/MLP_FSNet/results_seed2025.pkl

Files saved (or attempted):
  - model_seed2025.pt (model weights and architecture)
  - results_seed2025.pkl (training history, metrics, detailed test results)
Training and testing completed in 798.55 seconds
Done!!!


ORIGINAL README.md
-------------

This repository is by 
[Hoang T. Nguyen](https://www.linkedin.com/in/hoang-nguyen-971519201/) and 
[Priya L. Donti](https://www.priyadonti.com)
 and contains source code to reproduce the experiments in our paper 
 ["FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees"](https://arxiv.org/abs/2506.00362).


## Abstract
<p style="text-align: justify;">
Efficiently solving constrained optimization problems is crucial for numerous real-world applications, yet traditional solvers are often computationally prohibitive for real-time use. Machine learning-based approaches have emerged as a promising alternative to provide approximate solutions at faster speeds, but they struggle to strictly enforce constraints, leading to infeasible solutions in practice. To address this, we propose the Feasibility-Seeking-Integrated Neural Network (FSNet), which integrates a feasibility-seeking step directly into its solution procedure to ensure constraint satisfaction. This feasibility-seeking step solves an unconstrained optimization problem that minimizes constraint violations in a differentiable manner, enabling end-to-end training and providing guarantees on feasibility and convergence. Our experiments across a range of different optimization problems, including both smooth/nonsmooth and convex/nonconvex problems, demonstrate that FSNet can provide feasible solutions with solution quality comparable to (or in some cases better than) traditional solvers, at significantly faster speeds. 

<p align="center">
  <img src="figures\diagram.png" alt="FSNet Diagram" width="800"/>
</p>


If you find this repository helpful in your publications, please consider citing our paper.
```bash
@article{nguyen2025fsnet,
    title={FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees}, 
    author={Hoang T. Nguyen and Priya L. Donti},
    year={2025},
    journal={arXiv preprint arXiv:2506.00362},
}
```


## 🚀 Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎓 Usage

### Training and Test

```bash
python main.py \
  --method <FSNet|penalty|adaptive_penalty|DC3|projection> \
  --prob_type <convex|nonconvex|nonsmooth_nonconvex> \
  --prob_name <qp|qcqp|socp>
```

* `--method`

  * `FSNet`              (Feasibility-Seeking Neural Network)
  * `penalty`            (Penalty method)
  * `adaptive_penalty`   (Adaptive Penalty method)
  * `DC3`                (Deep Constraint Completion and Correction)
  * `projection`         (Projection-based method; supported for QP only)
* `--prob_type`

  * `convex`
  * `nonconvex`
  * `nonsmooth_nonconvex`
* `--prob_name`

  * `qp`   (Quadratic Program)
  * `qcqp` (Quadratically Constrained Quadratic Program)
  * `socp` (Second-Order Cone Program)
* And see `main.py` for more relevant flags.

Example:
```bash
python main.py --method FSNet --prob_type convex --prob_name qp
```
