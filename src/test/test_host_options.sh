#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

set -euo pipefail

# Usage: bash test_host_options.sh <build>/test <ranks> [MPI launcher and arguments]
# A HIP build can run this regression without GPUs. Adjacent flags must all be read.
test_dir=$(cd "${1:?Provide the directory containing struct, sstruct and sstruct.in.default}" && pwd)
ranks=${2:?Provide the MPI rank count}
[[ "$ranks" =~ ^[1-9][0-9]*$ ]]
shift 2
cd "$test_dir"
test -x ./struct && test -x ./sstruct && test -r ./sstruct.in.default

output=$(mktemp)
trap 'rm -f "$output"' EXIT

for driver in struct sstruct; do
   grid=(-P "$ranks" 1 1)
   if [ "$driver" = struct ]; then
      grid+=(-n 8 8 8)
   fi
   for option_order in memory_first execution_first memory_override execution_override device_options; do
      case "$option_order" in
         memory_first)
            options=(-memory_host -exec_host) ;;
         execution_first)
            options=(-exec_host -memory_host) ;;
         memory_override)
            options=(-memory_device -memory_host -exec_host) ;;
         execution_override)
            options=(-exec_device -exec_host -memory_host) ;;
         device_options)
            options=(-memory_host -lazy_device_init 1 -exec_host -device_id 0) ;;
      esac

      printf 'test=%s option_order=%s ranks=%s\n' "$driver" "$option_order" "$ranks"
      if ! "$@" "./$driver" "${options[@]}" "${grid[@]}" > "$output" 2>&1; then
         cat "$output"
         printf '%s %s: run failed\n' "$driver" "$option_order" >&2
         exit 1
      fi
      cat "$output"

      # Require a completed solve and a finite, converged residual as well as success.
      awk '
         /Iterations =/ { iterations = 1 }
         /Final Relative Residual Norm =/ {
            residual = $NF
            if (residual ~ /^[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$/ &&
                residual + 0 <= 1e-6) { converged = 1 }
         }
         END { exit !(iterations && converged) }
      ' "$output" || { cat "$output"; exit 1; }
      printf '%s %s: passed\n' "$driver" "$option_order"
   done
done
