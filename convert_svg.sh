for f in *.svg
do
    name=$(echo $f | cut -f 1 -d '.')
    inkscape -z -e ${name}.png -w 1024 -h 1024 $f
done
