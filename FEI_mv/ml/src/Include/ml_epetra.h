/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
