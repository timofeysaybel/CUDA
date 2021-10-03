build:
	nvcc src/*.cu src/filters/*.cu src/image/*.cpp src/io/*.cpp -lm -lpthread -lX11 -ljpeg -o ImageConvolution

clean:
	rm res/* ImageConvolution
