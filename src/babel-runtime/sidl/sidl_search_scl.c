/*
 * File:        sidl_search_scl.c
 * Copyright:   (c) 2003 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.8 $
 * Date:        $Date: 2007/09/27 19:35:48 $
 * Description: Search SCL files and directories
 *
 */

#include "sidl_search_scl.h"
#include <stdio.h>
#ifdef HAVE_LIBXML2
#include "sidl_String.h"
#include <dirent.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <libxml/SAX.h>
#include <libxml/parserInternals.h>

#define MAX_FILE_NAME_LEN 4096
static const char   g_SCL_FILE_EXT[] = sidl_SCL_EXT;
static const char   g_CCA_FILE_EXT[] = sidl_CCA_EXT;
static const size_t g_SCL_EXT_SIZE = sizeof(g_SCL_FILE_EXT);
static const size_t g_CCA_EXT_SIZE = sizeof(g_CCA_FILE_EXT);
static const size_t g_MIN_EXT_SIZE = 
  ((sizeof(g_SCL_FILE_EXT) < sizeof(g_CCA_FILE_EXT) )
    ? sizeof(g_SCL_FILE_EXT) : sizeof(g_CCA_FILE_EXT));

static void clear_entry(struct sidl_scl_entry *ent);

/**
 * Copy the directory name and ensure it ends with
 * the directory separator, '/'.
 * Return the adjusted length.
 */
static size_t
copyDirName(char *dest,
            const char *src,
            size_t dirlen)
{
  strcpy(dest, src);
  if (dest[dirlen-1] != '/') {
    dest[dirlen++] = '/';
    dest[dirlen] = '\0';
  }
  return dirlen;
}


static
struct sidl_scl_entry *
parse_file(const char *classname,
           const char *target,
           const char *filename);
           

/**
 * Search through a directory listing for files (not directories)
 * ending with g_SCL_FILE_EXT. These files are parsed looking
 * for matches to classname.
 */
static
struct sidl_scl_entry *
search_dir(const char *classname,
           const char *target,
           const char *dirname)
{
  struct sidl_scl_entry *result = NULL;
  size_t dirlen = strlen(dirname), namelen;
  if (dirlen < (MAX_FILE_NAME_LEN - g_MIN_EXT_SIZE - 1)) {
    char filename[MAX_FILE_NAME_LEN+1];
    DIR *d = opendir(dirname);
    struct stat filestat;
    struct dirent *ent;
    int sclExtIndex, ccaExtIndex;
    struct sidl_scl_entry *current;
    dirlen = copyDirName(filename, dirname, dirlen);
    if (d) {
      while ((ent = readdir(d))) {
        namelen = strlen(ent->d_name);
        if ((namelen + dirlen) < MAX_FILE_NAME_LEN) {
          sclExtIndex = 1 + namelen - g_SCL_EXT_SIZE;
          ccaExtIndex = 1 + namelen - g_CCA_EXT_SIZE;
          if (((sclExtIndex > 0) &&
               !strcmp(ent->d_name + sclExtIndex, g_SCL_FILE_EXT)) ||
              ((ccaExtIndex > 0) &&
               !strcmp(ent->d_name + ccaExtIndex, g_CCA_FILE_EXT))) {
            memcpy(filename + dirlen, ent->d_name, namelen+1);
            if (!stat(filename, &filestat) && !S_ISDIR(filestat.st_mode)) {
              current = parse_file(classname, target, filename);
              if (current) {
                if (result) {
                  sidl_scl_reportDuplicate(classname, current, result);
                }
                else {
                  result = current;
                }
              }
            }
          }
        }
      }
      closedir(d);
    }
  }
  return result;
}



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
sidl_search_scl(/* in  */ const char              *classname,
                /* in  */ const char              *target,
                /* in  */ const char              *filename)
{
  struct stat filestat;
  if (getenv("sidl_DEBUG_DLOPEN") || getenv("SIDL_DEBUG_DLOPEN")) {
    fprintf(stderr, "Searching for class %s, target %s, file %s\n",
	    classname, target, filename);
  }
  if (!stat(filename, &filestat)) {
    if (S_ISDIR(filestat.st_mode)) {
      return search_dir(classname, target, filename);
    }
    else {
      return parse_file(classname, target, filename);
    }
  }
  return NULL;
}

enum sidl_scl_states { 
  sclError,
  sclDocument,
  sclScl,
  sclLibrary,
  sclClass
};

struct sidl_xml_context {
  const char            *d_classname; /* class being sought */
  const char            *d_target;    /* target being sought */
  enum sidl_scl_states   d_state;
  struct sidl_scl_entry  d_ent;
  struct sidl_scl_entry *d_result;
};


static 
struct sidl_scl_entry *
copyEntity(const struct sidl_scl_entry *ent)
{
  struct sidl_scl_entry *result = malloc(sizeof(struct sidl_scl_entry));
  result->d_uri = sidl_String_strdup(ent->d_uri);
  result->d_scl = sidl_String_strdup(ent->d_scl);
  result->d_md5 = sidl_String_strdup(ent->d_md5);
  result->d_sha1 = sidl_String_strdup(ent->d_sha1);
  result->d_resolve = ent->d_resolve;
  result->d_scope = ent->d_scope;
  return result;
}

static void
storeLibraryAttributes(struct sidl_scl_entry *ent,
                       const xmlChar * const *attrs)
{
  if (attrs) {
    while (*attrs && *(attrs+1)) {
      if (!xmlStrcmp(*attrs, (const xmlChar *)"uri")) {
        if (ent->d_uri) free((void *)ent->d_uri);
        ent->d_uri = sidl_String_strdup((const char*) *(attrs+1));
      }
      else if (!xmlStrcmp(*attrs, (const xmlChar *) "scope")) {
        if (!xmlStrcmp(*(attrs+1), (const xmlChar *) "global")) {
          ent->d_scope = sidl_Scope_GLOBAL;
        }
        else {
          ent->d_scope = sidl_Scope_LOCAL;
        }
      }
      else if (!xmlStrcmp(*attrs, (const xmlChar *) "resolution")) {
        if (!xmlStrcmp(*(attrs+1), (const xmlChar *) "now")) {
          ent->d_resolve = sidl_Resolve_NOW;
        }
        else {
          ent->d_resolve = sidl_Resolve_LAZY;
        }
      }
      else if (!xmlStrcmp(*attrs, (const xmlChar *) "md5")) {
        if (ent->d_md5) free((void *)ent->d_md5);
        ent->d_md5 = sidl_String_strdup((const char*) *(attrs+1));
      }
      else if (!xmlStrcmp(*attrs, (const xmlChar *) "sha1")) {
        if (ent->d_sha1) free((void *)ent->d_sha1);
        ent->d_sha1 = sidl_String_strdup((const char*) *(attrs+1));
      }
      attrs += 2;
    }
  }
}

static int
classMatches(const char *classname,
             const char *target,
             const xmlChar * const *attrs)
{
  int result = 0;
  while (*attrs && *(attrs+1) && (result < 3)) {
    if (!xmlStrcmp(*attrs, (const xmlChar *)"name") &&
        !xmlStrcmp(*(attrs+1), (const xmlChar *)classname)) {
      result |= 1;
    }
    else if (!xmlStrcmp(*attrs, (const xmlChar *)"desc") &&
             !xmlStrcmp(*(attrs+1), (const xmlChar *)target)) {
      result |= 2;
    }
    attrs += 2;
  }
  return (result == 3);
}

static void 
sclStartElement(void *ctx, 
                const xmlChar *fullname,
                const xmlChar **attrs)
{
  struct sidl_xml_context *xmlCtx = (struct sidl_xml_context *)ctx;
  if (!xmlStrcmp(fullname, (const xmlChar *) "class")) {
    if (sclLibrary == xmlCtx->d_state) {
      xmlCtx->d_state = sclClass;
    }
    if ((sclClass == xmlCtx->d_state) && attrs) {
      if (classMatches(xmlCtx->d_classname, xmlCtx->d_target, attrs)) {
        if (xmlCtx->d_result) {
          sidl_scl_reportDuplicate(xmlCtx->d_classname,
                                   &(xmlCtx->d_ent),
                                   xmlCtx->d_result);
        }
        else {
          xmlCtx->d_result = copyEntity(&(xmlCtx->d_ent));
        }
      }
    }
    else {
      xmlCtx->d_state = sclError;
    }
  }
  else if (!xmlStrcmp(fullname, (const xmlChar *) "library")) {
    if (sclScl == xmlCtx->d_state) {
      clear_entry(&(xmlCtx->d_ent));
      storeLibraryAttributes(&(xmlCtx->d_ent), attrs);
      xmlCtx->d_state = sclLibrary;
    }
    else {
      xmlCtx->d_state = sclError;
    }
  }
  else if (!xmlStrcmp(fullname, (const xmlChar *) "scl")) {
    if (sclDocument == xmlCtx->d_state) {
      xmlCtx->d_state = sclScl;
    }
      else {
      xmlCtx->d_state = sclError;
    }
  }
  /* ignore other elements */
}

static void
sclEndElement(void *ctx,
              const xmlChar *name)
{
  struct sidl_xml_context *xmlCtx = (struct sidl_xml_context *)ctx;
  if (((xmlCtx->d_state == sclClass) ||
       (xmlCtx->d_state == sclLibrary)) &&
      !xmlStrcmp(name, (const xmlChar *)"library")) {
    xmlCtx->d_state = sclScl;
  }
  else if ((xmlCtx->d_state == sclScl) &&
           !xmlStrcmp(name, (const xmlChar *)"scl")) {
    xmlCtx->d_state = sclDocument;
  }
}

static void
sclCharacters(void          *ctx,
              const xmlChar *ch,
              int            len)
{
  struct sidl_xml_context *xmlCtx = (struct sidl_xml_context *)ctx;
  if ((sclScl == xmlCtx->d_state) ||
      (sclLibrary == xmlCtx->d_state) || 
      (sclClass == xmlCtx->d_state)) {
    while (len-- > 0) {
      if (!IS_BLANK(ch[len])) {
        fprintf(stderr, "Illegal characters found in SCL/CCA file.\n");
        xmlCtx->d_state = sclError;
        return;
      }
    }
  }
}

static
xmlEntityPtr
sclGetEntity(void *ignored,
             const xmlChar* name)
{
  return xmlGetPredefinedEntity(name);
}

static
void
sclStartDocument(void *user_data)
{
  struct sidl_xml_context *ctx = (struct sidl_xml_context *)user_data;
  ctx->d_ent.d_uri = ctx->d_ent.d_md5 = ctx->d_ent.d_sha1 = NULL;
  ctx->d_result = NULL;
  ctx->d_state = sclDocument;
}

static xmlSAXHandler s_SAXHandler  = {
  NULL,                         /* internalSubset */
  NULL,                         /* isStandable */
  NULL,                         /* hasInternalSubset */
  NULL,                         /* hasExternalSubset */
  NULL,                         /* resolveEntity */
  sclGetEntity,                 /* getEntity */
  NULL,                         /* entityDecl */
  NULL,                         /* notationDecl */
  NULL,                         /* attributeDecl */
  NULL,                         /* elementDecl */
  NULL,                         /* unparsedEntityDecl */
  NULL,                         /* setDocumentLocator */
  sclStartDocument,             /* startDocument */
  NULL,                         /* endDocument */
  sclStartElement,              /* startElement */
  sclEndElement,                /* endElement */
  NULL,                         /* reference */
  sclCharacters,                /* characters */
  NULL,                         /* ignorableWhitespace */
  NULL,                         /* processingInstructions */
  NULL,                         /* comment */
  NULL,                         /* warning */
  NULL,                         /* error */
  NULL,                         /* fatalError */
  NULL,                         /* getParameterEntity */
  NULL,                         /* cdataBlock */
  NULL                          /* externalSubset */
#if LIBXML2_VERSION >= 2004007
  , 1                             /* initialized */
#endif
};

static
struct sidl_scl_entry *
parse_file(const char *classname,
           const char *target,
           const char *filename)
{
  struct sidl_xml_context ctx;
  ctx.d_ent.d_scl = sidl_String_strdup(filename);
  ctx.d_classname = classname;
  ctx.d_target = target;
  sclStartDocument(&ctx);

  if (!xmlSAXUserParseFile(&s_SAXHandler, &ctx, filename)) {
  }

  /* cleanup */
  clear_entry(&(ctx.d_ent));
  free((void *)ctx.d_ent.d_scl);
  xmlCleanupParser();
  return ctx.d_result;
}
           
#else

struct sidl_scl_entry *
sidl_search_scl(/* in  */ const char              *classname,
                /* in  */ const char              *target,
                /* in  */ const char              *filename)
{
  fprintf(stderr, "\
Babel: Dynamic loading of classes disabled.\n\
Babel: Unable to locate class %s (target=%s).\n", classname, target); 
  return NULL;
}
#endif

static void clear_entry(struct sidl_scl_entry *ent)
{
  if (ent->d_uri) free((void *)ent->d_uri);
  if (ent->d_md5) free((void *)ent->d_md5);
  if (ent->d_sha1) free((void *)ent->d_sha1);
  ent->d_uri = ent->d_md5 = ent->d_sha1 = NULL;
}

void
sidl_destroy_scl(struct sidl_scl_entry *ent)
{
  if (ent) {
    clear_entry(ent);
    if (ent->d_scl) free((void *)ent->d_scl);
    ent->d_scl = NULL;
    free((void *)ent);
  }
}

void
sidl_scl_reportDuplicate(const char                  *classname,
                         const struct sidl_scl_entry *duplicate,
                         const struct sidl_scl_entry *first)
{
  fprintf(stderr, "\
Babel: Multiple libraries implement class '%s'\n\
Babel: First library: %s specified by %s\n\
Babel: Duplicate library: %s specified by %s\n",
          classname, first->d_uri, first->d_scl,
          duplicate->d_uri, duplicate->d_scl);
}

