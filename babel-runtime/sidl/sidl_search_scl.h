/*
 * File:        sidl_search_scl.h
 * Copyright:   (c) 2003 The Regents of the University of California
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Method to search an SCL file for a particular class
 *
 */

#ifndef included_sidl_search_scl_h
#define included_sidl_search_scl_h
#ifdef __cplusplus
extern "C" { /* } */
#endif
#ifndef included_sidl_Scope_IOR_h
#include "sidl_Scope_IOR.h"
#endif
#ifndef included_sidl_Resolve_IOR_h
#include "sidl_Resolve_IOR.h"
#endif

#define sidl_SCL_EXT ".scl"
#define sidl_CCA_EXT ".cca"

/**
 * The SCL entry contains 7 items of interest.
 * 1.   the uri/filename of the shared library
 * 2.   the uri/filename of the SCL file describing the shared library
 * 3.   an optional md5 checksum (as a hexidecimal number in plain text
 * 4.   an optional SHA1 checksum (as a hexidecimal number in plain text
 * 5.   whether the file should be loaded NOW or LAZY.
 * 6.   whether the file should be loaded local or global.
 * Optional entries may be NULL to indicate they aren't available.
 */
struct sidl_scl_entry {
  const char              *d_uri; /* uri or filename */
  const char              *d_scl; /* uri or filename for SCL file */
  const char              *d_md5; /* md5 checksum */
  const char              *d_sha1; /* sha1 checksum */
  enum sidl_Resolve__enum  d_resolve; /* now or lazy */
  enum sidl_Scope__enum    d_scope; /* local or global */
};

/**
 * Parse an SCL file or directory containing SCL files
 * and search for a particular class.
 * 
 * classname  the sidl type being sought
 * target     the type of entry being sought. Usually, this is
 *            "ior/impl" which searches for a shared library
 *            containing the IOR and implementation. "java" searches
 *            for Java client bindings. "python/impl" searches
 *            for the Python skeletons & impls.
 * filename   the SCL file's name or a directory name
 *
 * RETURN VALUE
 * A malloc'ed struct sidl_scl_entry or NULL to indicate
 * failure. The client is responsible for calling sidl_destroy_scl on
 * non-NULL return values.
 *
 * ISSUES
 * If an SCL file has more than one library entry that
 * provide instances of classname, this function will
 * issue a warning message for each duplicate and
 * return the first.
 */
struct sidl_scl_entry *
sidl_search_scl(/* in  */ const char *classname,
                /* in  */ const char *target,
                /* in  */ const char *filename);

/**
 * Deallocate all memory associated with ent.
 * Passing in NULL is safe.
 */
void
sidl_destroy_scl(struct sidl_scl_entry *ent);


/**
 * Report duplicate library providing a class to stderr.
 */
void
sidl_scl_reportDuplicate(const char                  *classname,
                         const struct sidl_scl_entry *duplicate,
                         const struct sidl_scl_entry *first);

#ifdef __cplusplus
}
#endif
#endif /*  included_sidl_search_scl_h */
