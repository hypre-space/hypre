#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


ex=ex9
dir=`basename \`pwd\``
keys=Aaamc

if [ "$dir" = "vis" ]; then
   dir=.
   mesh=$ex.mesh
   solu=$ex-u.sol
   solv=$ex-v.sol
else
   dir=vis
   mesh=vis/$ex.mesh
   solu=vis/$ex-u.sol
   solv=vis/$ex-v.sol
fi

if [ ! -e $mesh ]
then
   echo "Can't find visualization data for $ex!"
   exit
fi

echo "FiniteElementSpace" > $solu
echo "FiniteElementCollection: H1_2D_P1" >> $solu
echo "VDim: 1" >> $solu
echo "Ordering: 0" >> $solu
echo "" >> $solu
find $dir -name "$ex-u.sol.??????" | sort | xargs cat | sort | awk '{ print $2 }' >> $solu

glvis -m $mesh -g $solu -k $keys &

echo "FiniteElementSpace" > $solv
echo "FiniteElementCollection: H1_2D_P1" >> $solv
echo "VDim: 1" >> $solv
echo "Ordering: 0" >> $solv
echo "" >> $solv
find $dir -name "$ex-v.sol.??????" | sort | xargs cat | sort | awk '{ print $2 }' >> $solv

glvis -m $mesh -g $solv -k $keys
