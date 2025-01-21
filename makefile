
imageG=./assets/test_images/grayscale/test_image.png
imageC=./assets/test_images/rgb/rgb.png

frqci :
	echo "FRQCI encoding"
	./piaskownica/bin/geqie simulate --image $(imageC) --grayscale false --encoding frqci  --return-padded-counts true

frqi :
	echo "FRQI encoding"
	./piaskownica/bin/geqie simulate --image $(imageG) --encoding frqi   --return-padded-counts true

ncqi : 
	echo "NCQI encoding"
	./piaskownica/bin/geqie simulate --image $(imageG) --encoding NCQI  --return-padded-counts true

mcqi :
	echo "MCQI encoding"
	./piaskownica/bin/geqie simulate --image $(imageG) --encoding MCQI  --return-padded-counts true

