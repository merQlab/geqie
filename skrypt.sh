#!/bin/bash 

image=./assets/test_image.png

echo "FRQI encoding"
./piaskownica/bin/geqie simulate --image $image --encoding frqi 

# echo "NCQI encoding"
#./piaskownica/bin/geqie simulate --image $image --encoding ncqi 

#echo "MCQI encoding"
#./piaskownica/bin/geqie simulate --image $image --encoding mcqi 

echo "FRQCI encoding"
./piaskownica/bin/geqie simulate --image $image --encoding frqci 

