#! /usr/bin/env perl -p -i.bak
s/^#undef (.*)$/#ifndef \1\n#undef \1\n#endif/;
