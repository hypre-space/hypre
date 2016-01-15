#! /usr/bin/env python
from sys import argv
from os  import rename
from re  import compile, MULTILINE
malloc = compile("/\\*[^r*]*rpl_(malloc|realloc)[^*]*\\*/\\s+#undef (malloc|realloc)\\s+", MULTILINE)
undef = compile(r'^#undef\s+(.*)$', MULTILINE)
for file in argv[1:]:
    input = open(file, "r")
    content = input.read()
    input.close()
    input = None
    rename(file, file + ".bak")
    out = open(file, "w")
    content = malloc.sub("", content)
    out.write(undef.sub("#ifndef \\1\n#undef \\1\n#endif", content))
    out.close()
    content = None
    out = None
    
