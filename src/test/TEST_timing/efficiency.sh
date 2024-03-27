#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

#=============================================================================
# struct: timing and parallel scaling efficiency test
#=============================================================================

##float Time1 Time1A Time1B Time1C
##float Time8 Time8A Time8B Time8C
##float Time64 Time64A Time64B Time64C
##float Eff1 Eff2
#
# extract cpu clock times from solver phase, and calculate average
#
Time1A=$(grep "cpu clock time" efficiency.out.0 | tail -1 | awk '{print $6}')
Time1B=$(grep "cpu clock time" efficiency.out.1 | tail -1 | awk '{print $6}')
Time1C=$(grep "cpu clock time" efficiency.out.2 | tail -1 | awk '{print $6}')
((Time1 = (Time1A + Time1B Time1C) / 3.0))
Time8A=$(grep "cpu clock time" efficiency.out.3 | tail -1 | awk '{print $6}')
Time8B=$(grep "cpu clock time" efficiency.out.4 | tail -1 | awk '{print $6}')
Time8C=$(grep "cpu clock time" efficiency.out.5 | tail -1 | awk '{print $6}')
((Time8 = (Time8A + Time8B Time8C) / 3.0))
Time64A=$(grep "cpu clock time" efficiency.out.6 | tail -1 | awk '{print $6}')
Time64B=$(grep "cpu clock time" efficiency.out.7 | tail -1 | awk '{print $6}')
Time64C=$(grep "cpu clock time" efficiency.out.8 | tail -1 | awk '{print $6}')
((Time64 = (Time64A + Time64B Time64C) / 3.0))
#
# Calculate parallel scaling efficiency
#
##((Eff1 = Time1 / Time64))
##((Eff2 = Time8 / Time64))
Eff1=$(awk '{print $Time1/$Time64}')
Eff2=$(awk '{print $Time8/$Time64}')
awk '{if ($Eff1 < 0.30) print "Failure:T1/T64 is less than 30% ($Time1/$Time64=$Eff1)" > &2}'
awk '{if ($Eff2 < 0.90) print "Failure:T8/T64 is less than 90% ($Time8/$Time64=$Eff2)" > &2}'

echo "1 node results ($Time1A, $Time1B, $Time1C) Avg=$Time1" >> efficiency.log
echo "8 node results ($Time8A, $Time8B, $Time8C) Avg=$Time8" >> efficiency.log
echo "64 node results ($Time64A, $Time64B, $Time64C) Avg=$Time64" >> efficiency.log
echo "T1/T64=$Eff1 T8/T64=$Eff2" >> efficiency.log

#rm -f efficiency.out.0 efficiency.out.1 efficiency.out.2
#rm -f efficiency.out.3 efficiency.out.4 efficiency.out.5
#rm -f efficiency.out.6 efficiency.out.7 efficiency.out.8
