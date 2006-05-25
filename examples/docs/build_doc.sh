#!/bin/tcsh

set examplesdir = "$argv[1]"
set currentdir  = `pwd`

# Create the README_files directory
if (!(-d $examplesdir/README_files)) then
    mkdir $examplesdir/README_files
endif

# Syntax highlighting
cd $examplesdir/README_files
foreach target (`ls ../*.c`)
    $currentdir/code2html.perl -l c -n -o html $target $target.html
    mv $target.html .
end
foreach target (`ls ../*.f`)
    $currentdir/code2html.perl -l f -n -o html $target $target.html
    mv $target.html .
end
foreach target (`ls ../*.cxx`)
    $currentdir/code2html.perl -l c++ -n -o html $target $target.html
    mv $target.html .
end
cd $currentdir

# Copy the example files
foreach file (`ls ex*.htm`)
    cp -fp "$file" "$file"l
end

# Replace the server side includes
foreach file (`ls *.htm`)
    $currentdir/replace-ssi.perl "$file" > $examplesdir/README_files/"$file"l
end

# Copy images
cp -fp *.gif *.jpg $examplesdir/README_files

# Remove the html example files
rm -f ex*.html

# Rename index.html
mv $examplesdir/README_files/index.html $examplesdir/README.html
