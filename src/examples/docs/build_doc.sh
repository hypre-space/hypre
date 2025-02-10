#!/bin/bash

examplesdir="$1"
currentdir=`pwd`

# Create the README_files directory
if [ ! -d $examplesdir/README_files ]; then
    mkdir $examplesdir/README_files
fi

# Syntax highlighting
cd $examplesdir/README_files
for target in `ls ../*.c`; do
    $currentdir/code2html.perl -l c -n -o html $target $target.html
    mv $target.html .
done
for target in `ls ../*.f*`; do
    $currentdir/code2html.perl -l f -n -o html $target $target.html
    mv $target.html .
done
for target in `ls ../*.cxx`; do
    $currentdir/code2html.perl -l c++ -n -o html $target $target.html
    mv $target.html .
done
cd $currentdir

# Copy the example files
for file in `ls ex*.htm`; do
    cp -fp "$file" "$file"l
done

# Replace the server side includes
for file in `ls *.htm`; do
    $currentdir/replace-ssi.perl "$file" > $examplesdir/README_files/"$file"l
done

# Copy images
cp -fp *.gif $examplesdir/README_files

# Remove the html example files
rm -f ex*.html

# Rename index.html
mv $examplesdir/README_files/index.html $examplesdir/README.html
