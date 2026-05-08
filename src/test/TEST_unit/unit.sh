#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# Check that all tests passed
#=============================================================================

# Find all output files
OUTFILES=$(ls ${TNAME}.out.* 2>/dev/null | sort)

if [ -z "$OUTFILES" ]; then
  echo "No output files found matching ${TNAME}.out.*" >&2
  exit 1
fi

# Build mapping from output file to test driver from jobs file
declare -A DRIVER_MAP
if [ -f ${TNAME}.jobs ]; then
  while IFS= read -r line; do
    # Extract lines like: mpirun -np 8  ./test_csr_overlap > unit.out.0
    if echo "$line" | grep -qE ">\s*${TNAME}\.out\.[0-9]+"; then
      DRIVER=$(echo "$line" | grep -oE "\./[a-zA-Z0-9_]+" | sed 's|^\./||')
      OUTFILE=$(echo "$line" | grep -oE "${TNAME}\.out\.[0-9]+" | head -1)
      if [ -n "$DRIVER" ] && [ -n "$OUTFILE" ]; then
        DRIVER_MAP[$OUTFILE]=$DRIVER
      fi
    fi
  done < ${TNAME}.jobs
fi

# Check each output file for failures
TOTAL_PASSED=0
TOTAL_FAILED=0
FAILED_DETAILS=""

for OUTFILE in $OUTFILES; do
  if [ ! -f $OUTFILE ]; then
    continue
  fi
  
  # Count PASSED and FAILED messages
  PASSED_COUNT=$(grep -c "PASSED" $OUTFILE 2>/dev/null || echo "0")
  FAILED_COUNT=$(grep -c "FAILED" $OUTFILE 2>/dev/null || echo "0")
  
  # Strip whitespace
  PASSED_COUNT=$(echo "$PASSED_COUNT" | tr -d '[:space:]')
  FAILED_COUNT=$(echo "$FAILED_COUNT" | tr -d '[:space:]')
  PASSED_COUNT=${PASSED_COUNT:-0}
  FAILED_COUNT=${FAILED_COUNT:-0}
  
  TOTAL_PASSED=$((TOTAL_PASSED + PASSED_COUNT))
  TOTAL_FAILED=$((TOTAL_FAILED + FAILED_COUNT))
  
  if [ "$FAILED_COUNT" -gt 0 ]; then
    # Find which test(s) failed in this file
    FAILED_TESTS=$(grep "^Test[0-9]" $OUTFILE | grep -v "PASSED" | sed 's/ (.*procs).*//' | sed 's/:.*//')
    
    # Get test driver for this output file
    TEST_DRIVER="${DRIVER_MAP[$OUTFILE]}"
    if [ -z "$TEST_DRIVER" ]; then
      TEST_DRIVER="Unknown"
    fi
    
    # Add to failed details
    if [ -n "$FAILED_TESTS" ]; then
      while IFS= read -r test; do
        FAILED_DETAILS="${FAILED_DETAILS}  - [${TEST_DRIVER}]: ${test}\n"
      done <<< "$FAILED_TESTS"
    fi
  fi
done

# Report failures if any
if [ "$TOTAL_FAILED" -gt 0 ]; then
  echo "Failed test summary:" >&2
  echo -e "$FAILED_DETAILS" | sed '/^$/d' >&2
fi

if [ "$TOTAL_PASSED" -eq 0 ] && [ "$TOTAL_FAILED" -eq 0 ]; then
  echo "No test results found in output files" >&2
fi

# Collect summary output from all output files
{
  for OUTFILE in $OUTFILES; do
    if [ -f $OUTFILE ]; then
      echo -e "# Output file: $OUTFILE\n"
      # Extract test results (lines containing PASSED/FAILED)
      grep -E "(PASSED|FAILED|Test[0-9])" $OUTFILE | head -20
      echo ""
    fi
  done
} > ${TNAME}.out

# Verify we got results
OUTCOUNT=$(grep -c "PASSED\|FAILED" ${TNAME}.out 2>/dev/null || echo "0")
OUTCOUNT=$(echo "$OUTCOUNT" | tr -d '[:space:]')
OUTCOUNT=${OUTCOUNT:-0}
if [ "$OUTCOUNT" -eq 0 ]; then
   echo "No test results found in output files" >&2
fi


