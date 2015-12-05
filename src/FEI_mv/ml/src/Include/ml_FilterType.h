/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
 ***********************************************************************EHEADER*/




#ifndef ML_FILTERTYPE_H
#define ML_FILTERTYPE_H

/* \file ml_FilterType.h
 *
 * \brief Enum for filtering.
 */

namespace ML_Epetra {

/*! \enum FilterType
 * 
 * \brief Defined the type of filter to be applied after each
 *  ExtractMyRowCopy().
 * 
 * \author Marzio Sala, SNL 9214.
 *
 * \date Last updated on 15-Mar-05.
 */

enum FilterType {
  ML_NO_FILTER,           /*< no filter is applied */
  ML_EQN_FILTER,          /*< decouples the equations */
  ML_TWO_BLOCKS_FILTER,   /*< decoupled the system in two blocks */
  ML_THREE_BLOCKS_FILTER, /*< decoupled the system in three blocks */
  ML_MASK_FILTER          /*< general approach */
};

} // namespace ML_Epetra
#endif
