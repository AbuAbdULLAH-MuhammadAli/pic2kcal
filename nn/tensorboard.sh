#!/bin/bash
cd "$(dirname "$0")"
mkdir -p "/tmp/$(whoami)"
TMPDIR="/tmp/$(whoami)" tensorboard --logdir runs "$@"
