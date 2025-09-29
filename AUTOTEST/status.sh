#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Store the exit code of the command that ran *before* this script.
# It's important to capture this early before other commands change its value.
PREVIOUS_EXIT_CODE=$?

# --- Main Logic ---
# Check if a testname was passed as the first argument ($1).
if [ -n "$1" ]; then
    testname="$1"
    err_file="${testname}.err"

    # Check if the corresponding error file exists.
    if [ -e "$err_file" ]; then
        # The file exists, now check if it's empty (has zero size).
        # The '-s' test returns true if the file has a size greater than zero.
        if [ -s "$err_file" ]; then
            echo -en "FAILED!"
            exit 1
        else
            echo -en "PASSED!"
            exit 0
        fi
    fi
fi

# --- Fallback Logic ---
# If we are here, it means one of two things:
# 1. No testname was provided as an argument.
# 2. A testname was provided, but the corresponding .err file was not found.
# In either case, we fall back to checking the exit code of the previous command.

if [ "$PREVIOUS_EXIT_CODE" -eq 0 ]; then
    echo -en "PASSED!"
    exit 0
else
    echo -en "FAILED!"
    exit 1
fi
