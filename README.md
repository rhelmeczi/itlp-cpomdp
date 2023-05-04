# ITLP for CPOMDP

This repository provides some sample code from my paper on constrained POMDPs,
which can be read [here](https://arxiv.org/pdf/2206.14081.pdf).
Specifically, I provide a simplified implementation for our algorithm used to solve 
constrained POMDP problems in a finite horizon setting.

# Usage

Install the requirements:

```
pip install -r requirements.txt
```

and then use the code via the commandline:

```
python cli.py data_files/tiger_problem/
```

## Gurobi

To solve the underlying LP models, we use gurobi as it is free for academics
and is a leading LP solver in practice. Unfortunately, this is proprietary software,
and a license must be acquired to use this code in practice. To use this code
for yourself, see their [website](https://www.gurobi.com/downloads/).
