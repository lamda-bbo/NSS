# Multi-objective Optimization-based Selection for Quality-Diversity by Non-surrounded-dominated Sorting

We use the code of [DQD-RL](https://github.com/icaros-usc/dqd-rl) as the framework for QDGym.

## Preparation

First, install [singularity](https://apptainer.org).

Then, build the container:

```sh
sudo make container.sif
```

## Running Experiments

To run the experiments on a local machine:
```sh
bash script/run_local.sh CONFIG SEED NUM_WORKERS
```
where the `CONFIG` is a gin file in [config/](config/).

For more information see [README-dqdrl.md](README-dqdrl.md)
