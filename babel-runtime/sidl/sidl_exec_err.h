/*
 * File:        sidl_exec_err.h
 * Copyright:   (c) 2001-2003 The Regents of the University of California
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: convenience C macros for managing sidl exceptions in exec functions
 *
 * See sidl_Exception.h for more information on handling exception in Babel.
 * This file is specifically for a special macro, only used in IOR.c files,
 * that helps exec functions handle exceptions. 
 *
 * Copyright (c) 2000-2001, The Regents of the University of Calfornia.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * UCRL-CODE-2002-054
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#ifndef included_sidl_exec_err_h
#define included_sidl_exec_err_h

#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif

#ifndef NULL
#define NULL 0
#endif

/*
 * Define __FUNC__ to be "unknown" so that we do not force users to define
 * __FUNC__ before their functions.
 */

#ifndef __FUNC__
#define __FUNC__ "unknown"
#endif

/**
 * sidl helper macro that checks the status of an exception.  If the exception
 * is not set, then this macro does nothing.  If the exception is set, then
 * a stack trace line is added to the exception and control jumps to the user
 * defined EXIT block for exception processing.
 *
 * Suggested usage: This macro should be placed at the end of the line of
 * each function call.  By doing so, the line entered into the stack trace
 * is more accurate and the code more readable.
 *
 * EXAMPLE:
 * void myfunction(..., sidl_BaseInterface *_ex)
 * {
 *   ...
 *   foo(..., _ex); EXEC_CHECK(*_ex);
 *   ...
 *   EXEC_ERR:;
 *     / * clean up and return with exception set in _ex * /
 * }
 *
 * WARNINGS:  
 * Do not use this within an EXEC_ERR block!
 */
#define EXEC_CHECK(EX_VAR) {						                     \
  if ((EX_VAR) != NULL) {							             \
    sidl_BaseInterface _throwaway_exception = NULL;                                          \
    sidl_BaseException _exec_ex = sidl_BaseException__cast((EX_VAR), &_throwaway_exception); \
    sidl_BaseException_add(_exec_ex, __FILE__, __LINE__, __FUNC__, &_throwaway_exception);   \
    sidl_BaseException_deleteRef(_exec_ex, &_throwaway_exception);                           \
    goto EXEC_ERR;							                     \
  }									                     \
}

#endif
