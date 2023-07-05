# Multi-objective Optimization-based Selection for Quality-Diversity by Non-surrounded-dominated Sorting

We use the code of [DSAGE](https://github.com/icaros-usc/dsage) as the framework for Mario environment generation.

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
where the `CONFIG` is a gin file in [config/mario/](config/mario/).

For more information see [README-dsage.md](README-dsage.md)
