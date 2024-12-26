
image=./assets/test_image.png

frqci :
	echo "FRQCI encoding"
	./piaskownica/bin/geqie simulate --image $(image) --encoding frqci 

frqi :
	echo "FRQI encoding"
	./piaskownica/bin/geqie simulate --image $(image) --encoding frqi 

ncqi : 
	echo "NCQI encoding"
	./piaskownica/bin/geqie simulate --image $(image) --encoding NCQI

mcqi :
	echo "MCQI encoding"
	./piaskownica/bin/geqie simulate --image $(image) --encoding MCQI

