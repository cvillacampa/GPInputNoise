# GPInputNoise

Multi-class Gaussian Process Classification with Noisy Inputs

Requirements: Python2.7, dill0.2.9, tensorflow1.15.2 and numpy1.16

You can install all the Python dependencies needed to execute the experiments with pip by running:

``` bash
pip install -r requirements.txt

```

## Experiments

This repository comes with a series of experiments to evaluate the performance of the proposed approaches:
 * Synthetic: 100 randomly generated 2D synthetic datasets.
 * UCI: 8 multi-class datasets from the UCI repository.
 * MNIST: The MNIST dataset of handwritten digits.
 * Atrophysics: A real-world dataset from astrophysics.
 * Active learning: An active learning problem.

### Experimental data

In order to run the experiments, the first step is to generate the data and/or the splits. For that, uncompress the file data.tgz contained in both synthetic and astrophysics data folders and run the script generate_data.py for each dataset in the uci folder. For the MNIST dataset, go to experiments/mnist/data and run the script join_data.sh

### Running the experiments

To run any of the experiments, go to the corresponding folder and run either test_real.py for the UCI, MNIST and astrophysics datasets or test_toy.py for the synthetic datasets. In all the cases but MNIST the launching script receives the split as an argument.

- Examples:

```bash
> cd experiments/synthetic/noise_0.1/MGPC
> python2 test_toy.py 0
```

```bash
> cd experiments/uci/glass/0.1/MGPC
> python2 test_real.py 0
```

```bash
> cd experiments/mnist/0.1_gpu/MGPC
> python2 test_real.py
```

```bash
> cd experiments/astrophysics/NIMGPC
> python2 test_real.py 0
```

```bash
> cd experiments/active_learning/0.1/NIMGPC
> python2 test_real.py 0 min_ll
```
