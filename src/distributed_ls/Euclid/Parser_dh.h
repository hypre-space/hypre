/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef PARSER_DH_DH
#define PARSER_DH_DH

/* #include "euclid_common.h" */

extern void Parser_dhCreate(Parser_dh *p);
extern void Parser_dhDestroy(Parser_dh p);

extern bool Parser_dhHasSwitch(Parser_dh p, const char *in);
extern bool Parser_dhReadString(Parser_dh p, const char *in, char **out);
extern bool Parser_dhReadInt(Parser_dh p, const char *in, HYPRE_Int *out);
extern bool Parser_dhReadDouble(Parser_dh p, const char *in, HYPRE_Real *out);
  /* if the flag (char *in) is found, these four return
     true and set "out" accordingly.  If not found, they return
     false, and "out" is unaltered.
   */

extern void Parser_dhPrint(Parser_dh p, FILE *fp, bool allPrint);
  /* Outputs all <flag,value> pairs.  "bool allPrint" is
   * only meaningful when Euclid is compiled in MPI mode
   */

extern void Parser_dhInsert(Parser_dh p, const char *name, const char *value);
  /* For inserting a new <flag,value> pair, or altering
   * the value of an existing pair from within user apps.
   */

extern void Parser_dhUpdateFromFile(Parser_dh p, const char *name);

extern void Parser_dhInit(Parser_dh p, HYPRE_Int argc, char *argv[]);
  /* Init enters <flag,value> pairs in its internal database in
     the following order:

       (1)   $PCPACK_DIR/options_database
       (2)   "database" in local directory, if the file exists
       (3)   "pathname/foo" if argv[] contains a pair of entries:
               -db_filename pathname/foo
       (4)   flag,value pairs from the command line (ie, argv)

      If a flag already exists, its value is updated if it is
      encountered a second time.

      WARNING! to enter a negative value, you must use two dashes, e.g:
                      -myvalue  --0.1
               otherwise, if you code "-myvalue -0.1" you will have entered
               the pair of entries <-myvalue, 1>, <-0.1, 1>.  Yuck!@#
               But this works, since Euclid doesn't use negative numbers much.

      If the 2nd entry is missing, a value of "1" is assumed (this only
      works on the command line; for files, you must explicitly code a
      value.  See $PCPACK_DIR/options_database for samples.

      The following will cause Parser_dhHasSwitch(p, "-phoo") to return false:
          -phoo 0
          -phoo false
          -phoo False
          -phoo FALSE
      any other value, including something silly like -phoo 0.0
      will return true.
   */

#endif
