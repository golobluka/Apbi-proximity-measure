# APBI Proximity Measure

This repository contains code related to work on multiobjective optimization at the Jožef Stefan Institute. The theoretical background for this work is derived from the scientific paper:

> Mohammed Jameel and Mohamed Abouhawwash: A new proximity metric based on optimality conditions for single and multi-objective optimization: Method and validation

**Implementation:**

I have implemented two proximity measures, which directly correspond to the paper:

* **PBI Proximity Measure:** Implemented in the file `pbi_proximity_measure.py`.
* **APBI Proximity Measure:** Implemented in the file `apbi_proximity_measure.py`.

**Notebooks:**

The repository includes two Jupyter notebooks that demonstrate the usage of these measures on optimization problems defined in `optimization_problems.py`. These optimization problems are primarily sourced from standard benchmarks implemented through the `pymoo` library.

* **`running_nsga_and_plotting.ipynb`:** This notebook executes the NSGA-II algorithm and plots the implemented proximity measures to evaluate their expected behavior.
* **`checking_proximity_measure_values.ipynb`:** This notebook aims to reproduce the tables from the aforementioned paper, showcasing specific evaluations of the PBI and APBI proximity measures.

**Current Findings:**

In the last cell of `checking_proximity_measure_values.ipynb`, an error persists. Furthermore, the calculated values from our implementation differ considerably from the results presented in the paper. This discrepancy might indicate a potential error in the underlying methodology or implementation that requires further investigation.
