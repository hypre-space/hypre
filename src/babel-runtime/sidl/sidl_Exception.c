/*
 * File:        sidl_Exception.c
 * Copyright:   (c) 2005 The Regents of the University of California
 * Release:     $Name:  $
 * Revision:    @(#) $Revision: 1.1 $
 * Date:        $Date: 2006/08/29 23:31:42 $
 * Description: 
 *
 */
#include "sidl_Exception.h"
#include "sidl_BaseInterface.h"
#include "sidl_BaseException.h"

int
SIDL_CATCH(struct sidl_BaseInterface__object *ex_var,
           const char *sidl_Name)
{
  struct sidl_BaseInterface__object *exception=NULL, *throwaway_exception;
  const int result = 
    (ex_var && sidl_BaseInterface_isType(ex_var, sidl_Name, &exception) &&
     !exception);
  if (exception) sidl_BaseInterface_deleteRef(exception, &throwaway_exception);
  return result;
}

void
sidl_update_exception(struct sidl_BaseInterface__object *ex,
                      const char *filename,
                      const int32_t line,
                      const char *funcname)
{
  sidl_BaseInterface exception = NULL; 
  sidl_BaseException _s_b_e = sidl_BaseException__cast(ex, &exception);
  sidl_BaseException_add(_s_b_e, filename, line, funcname, &exception);
  sidl_BaseException_deleteRef(_s_b_e, &exception); 
}
