#ifndef isisfei_h_included
#define isisfei_h_included
/*
          This file is part of Sandia National Laboratories
          copyrighted software.  You are legally liable for any
          unauthorized use of this software.

          NOTICE:  The United States Government has granted for
          itself and others acting on its behalf a paid-up,
          nonexclusive, irrevocable worldwide license in this
          data to reproduce, prepare derivative works, and
          perform publicly and display publicly.  Beginning five
          (5) years after June 5, 1997, the United States
          Government is granted for itself and others acting on
          its behalf a paid-up, nonexclusive, irrevocable
          worldwide license in this data to reproduce, prepare
          derivative works, distribute copies to the public,
          perform publicly and display publicly, and to permit
          others to do so.

          NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED
          STATES DEPARTMENT OF ENERGY, NOR SANDIA CORPORATION,
          NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS
          OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR
          RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
          USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT, OR
          PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
          INFRINGE PRIVATELY OWNED RIGHTS.

Latest released version: 1.1.0

*/
//requires:
//#include <stdlib.h>
//#include <math.h>

#ifdef FEI_SER
#include <mpiuni/mpi.h>
#include <isis-ser.h>
#else
#include <mpi.h>
#include <isis-mpi.h>
#endif

#include "other/basicTypes.h"
#include "fei.h"
#include "src/CommBufferDouble.h"
#include "src/CommBufferInt.h"
#include "src/NodePackets.h"
#include "src/BCRecord.h"
#include "src/FieldRecord.h"
#include "src/BlockRecord.h"
#include "src/MultConstRecord.h"
#include "src/PenConstRecord.h"
#include "src/SimpleList.h"
#include "src/NodeRecord.h"
#include "src/SharedNodeRecord.h"
#include "src/SharedNodeBuffer.h"
#include "src/ExternNodeRecord.h"
#include "src/SLE_utils.h"

#include "src/BASE_SLE.h"
#include "src/ISIS_SLE.h"
#include "src/FEI_ISIS.h"

#endif /*isisfei_h_included*/
