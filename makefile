
image=./assets/test_image.png

frqci :
	echo "FRQCI encoding"
	./piaskownica/bin/geqie simulate --image $(image) --grayscale false --encoding frqci  --return-padded-counts true

frqi :
	echo "FRQI encoding"
	./piaskownica/bin/geqie simulate --image $(image) --encoding frqi   --return-padded-counts true

ncqi : 
	echo "NCQI encoding"
	./piaskownica/bin/geqie simulate --image $(image) --encoding NCQI  --return-padded-counts true

mcqi :
	echo "MCQI encoding"
	./piaskownica/bin/geqie simulate --image $(image) --encoding MCQI  --return-padded-counts true

