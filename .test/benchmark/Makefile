
CC=nvcc
CFLAGS=-Xcompiler -fopenmp
FLAGS=-arch=sm_30 -O3
PATHS=-I ../../include -I ../boost

all: intlag

intlag: main.cu *.h ../../include
	$(CC) $(PATHS) $(FLAGS) $(CFLAGS) main.cu -o intlag

clean:
	rm -rf *.o intlag
