# Multi-objective Optimization-based Selection for Quality-Diversity by Non-surrounded-dominated Sorting

We use the code of [DQD](https://github.com/icaros-usc/dqd) as the framework for Robotic arm.

## Preparation

First, create the conda environment:

```sh
conda env create -f experiments/environment.yml
```

Next install the local copy of pyribs after activating conda:

```bash
conda activate dqdexps
pip3 install -e .[all]
```

## Running Experiments

To run the experiments:
```sh
cd experiments/arm
python3 arm.py og_map_elites METHOD
```
where the `METHOD` can be `random`, `mop1`, `mop2`, `mop3`, `nslc`, `curiosity`, `edocs`, or `nss`

For more information see [README-dqd.md](README-dqd.md)
