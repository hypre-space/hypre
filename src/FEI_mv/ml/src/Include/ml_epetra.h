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




#ifndef ML_EPETRA_H
#define ML_EPETRA_H

// prints out an error message if variable is not zero,
// and return this value. This macro always returns.
#define ML_RETURN(ml_err) \
  { if (ml_err != 0) { \
    cerr << "ML::ERROR:: " << ml_err << ", " \
      << __FILE__ << ", line " << __LINE__ << endl; } \
      return(ml_err);  } 

// prints out an error message if variable is not zero,
// and return this value.
#define ML_CHK_ERR(ml_err) \
  { if (ml_err != 0) { \
    cerr << "ML::ERROR:: " << ml_err << ", " \
      << __FILE__ << ", line " << __LINE__ << endl; \
      return(ml_err);  } }

// prints out an error message if variable is not zero
// and returns.
#define ML_CHK_ERRV(ml_err) \
  { if (ml_err != 0) { \
    cerr << "ML::ERROR:: " << ml_err << ", " \
      << __FILE__ << ", line " << __LINE__ << endl; \
    return; } }

#define ML_EXIT(ml_err) \
  { if (ml_err != 0) { \
    cerr << "ML::FATAL ERROR:: " << ml_err << ", " \
      << __FILE__ << ", line " << __LINE__ << endl; } \
    exit(ml_err); }

#endif
