## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.0001 | 37.34%   | 0.00%    |
| 0.001  | 42.96%   | 1.69%    |
| 0.01   | 35.16%   | 100.00%  |

## Analysis

The results clearly demonstrate the impact of lambda on sparsity:

- At λ = 0.0001, the sparsity penalty is too weak, so no pruning occurs.
- At λ = 0.001, the model begins to prune some connections while maintaining reasonable accuracy.
- At λ = 0.01, the sparsity penalty becomes too strong, causing all gates to collapse to zero, leading to complete pruning and reduced performance.

This shows that selecting an appropriate lambda is critical to balance model efficiency and accuracy.
## Gate Distribution Analysis

For λ = 0.01, the gate distribution shows a large spike at zero, indicating that almost all connections are pruned.

This confirms that a strong L1 penalty forces the model to aggressively eliminate weights, sometimes excessively, leading to loss of useful information.

## Conclusion

This project successfully demonstrates a self-pruning neural network using learnable gates and L1 regularization.

The network dynamically removes less important weights during training. However, the degree of pruning is highly dependent on the regularization strength.

An optimal lambda value is necessary to achieve a balance between sparsity and model performance.
