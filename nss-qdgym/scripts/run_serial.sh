#!/bin/bash
# Runs experiments serially on a local computer.
#
# Usage:
#   bash scripts/run_serial.sh CONFIG SEED [RELOAD_PATH]

print_header() {
  echo
  echo "------------- $1 -------------"
}

# Prints "=" across an entire line.
print_thick_line() {
  printf "%0.s=" $(seq 1 `tput cols`)
  echo
}

#
# Parse command line flags.
#

CONFIG="$1"
SEED="$2"
RELOAD_PATH="$3"
if [ -z "${SEED}" ]
then
  echo "Usage: bash scripts/run_serial.sh CONFIG SEED [RELOAD_PATH]"
  exit 1
fi

if [ -n "$RELOAD_PATH" ]; then
  RELOAD_ARG="--reload-dir ${RELOAD_PATH}"
else
  RELOAD_ARG=""
fi

set -u  # Uninitialized vars are error.

#
# Run the experiment.
#

print_header "Running experiment"
echo
print_thick_line
singularity exec --cleanenv --nv container.sif \
  python -m src.main \
    --config "$CONFIG" \
    --address "0" \
    --seed "$SEED" \
    $RELOAD_ARG
print_thick_line
