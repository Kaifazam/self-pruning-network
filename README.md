#  Self-Pruning Neural Network

##  Overview

This project implements a **self-pruning neural network** using PyTorch.
Unlike traditional pruning (done after training), this model **learns to prune itself during training** using learnable gate parameters.

Each weight is associated with a gate value (0 to 1).

* Gate ≈ 1 → weight is active
* Gate ≈ 0 → weight is pruned

---

## Key Idea

The model introduces a custom layer called **PrunableLinear**, where:

* Each weight has a corresponding **gate score**
* A **sigmoid function** converts gate scores into values between 0 and 1
* Final weight = weight × gate

To encourage pruning, we add an **L1 sparsity loss**:

Total Loss = Classification Loss + λ × Sparsity Loss

---

##  Technologies Used

* Python
* PyTorch
* Torchvision
* Matplotlib

---

## Project Structure

```
self-pruning-network/
│
├── model.py        # Custom prunable layer and model
├── utils.py        # Sparsity loss, evaluation functions
├── train.py        # Training and testing script
├── report.md       # Case study report
├── requirements.txt
└── README.md
```

---

## Installation & Setup

1. Clone the repository:

```
git clone <your-repo-link>
cd self-pruning-network
```

2. Create virtual environment:

```
python -m venv venv
venv\Scripts\activate   # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

##  How to Run

Run the training script:

```
python train.py
```

---

##  Experiments

We tested the model with different values of λ (lambda):

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 0.0001 | 37.34%   | 0.00%    |
| 0.001  | 42.96%   | 1.69%    |
| 0.01   | 35.16%   | 100.00%  |

---

##  Observations

* Low λ → High accuracy, no pruning
* Medium λ → Balanced pruning
* High λ → High sparsity but low accuracy

 This shows the **trade-off between performance and efficiency**

---

##  Gate Distribution

The histogram of gate values shows how weights are pruned:

* Low λ → gates spread out (no pruning)
* High λ → spike at 0 (heavy pruning)

---

##  Conclusion

This project demonstrates that:

* Neural networks can **learn to prune themselves**
* L1 regularization encourages sparsity
* Choosing the right λ is critical

---

##  Future Improvements

* Use CNN instead of fully connected layers
* Improve accuracy with better architecture
* Apply structured pruning techniques

---

##  Author

Kaif

---

##  Note

A very high λ can lead to **over-pruning**, where all weights are removed, reducing model performance.
