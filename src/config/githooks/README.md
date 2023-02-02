<!--
Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
HYPRE Project Developers. See the top-level COPYRIGHT file for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)
-->

This directory contains recommended git hooks for hypre:

### The hooks (currently only one)

* `pre-commit` is a hook that is applied before each commit that runs `astyle`
to format code according to hypre coding style guidelines.

### Setup

To setup the git hooks, copy the hooks to the `.git/hooks` directory (or create
a symbolic link to them, e.g., `ln -s ../../src/config/githooks/pre-commit .`).

