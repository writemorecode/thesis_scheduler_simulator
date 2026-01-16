### Multidimensional Heterogeneous First-Fit Bin Packing

This package provides scheduling and packing algorithms for heterogeneous
multidimensional bin packing. Use the public API to run a single problem
instance on any supported scheduler.

#### Quick Start

```python
import numpy as np

from simulator import ProblemInstance, run_instance

problem = ProblemInstance(
    capacities=np.array([[8, 12], [4, 6]]),
    requirements=np.array([[3, 5], [2, 1]]),
    job_counts=np.array([4, 3]),
    purchase_costs=np.array([10.0, 14.0]),
    running_costs=np.array([1.0, 1.5]),
    resource_weights=np.array([1.0, 1.0]),
)

result = run_instance(problem, scheduler="ffd")
print(result.total_cost)
```

Install the exact ILP solver extras when you need `solve_exact`:

```bash
uv add "simulator[exact]"
```

#### Evaluation Workflow

Dataset generation and evaluation scripts now live in the thesis repository
(`thesis/scripts`). Run the evaluation workflow from that project so the
dependencies stay separated from the simulator library.
