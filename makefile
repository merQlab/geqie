
image=./assets/test_image.png

efrqi:
	echo "EFRQI encoding"
	./piaskownica/bin/geqie simulate --image $(image)  --encoding efrqi

frqi :
	echo "FRQI encoding"
	./piaskownica/bin/geqie simulate --image $(image) --encoding frqi 

