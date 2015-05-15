
CC=nvcc -ccbin /usr/bin/g++-4.8
FLAGS=-arch=sm_30 -lgmp -lmpfr
PTHREAD=-Xcompiler -pthread
GTEST=-Xcompiler -pthread -lgtest
PATHS=-I ../include -I ../gtest-1.7.0/include -L ../gtest-1.7.0/build -I . -L/usr/local/cuda/lib64 -I/usr/local/cuda/include

all: test


test: device_data_test.o rounder_test.o reference.o interval_test.o box_test.o cuda_interval_test.o cuda_box_test.o serial_blas1_test.o serial_blas2_test.o serial_blas3_test.o omp_blas1_test.o omp_blas2_test.o omp_blas3_test.o cuda_blas1_test.o cuda_blas2_test.o cuda_blas3_test.o main.cu
	$(CC) $(PATHS) $(FLAGS) $(GTEST) *.o main.cu -o test



reference.o: ../include aux/reference.h aux/reference.cu
	$(CC) $(PATHS) $(FLAGS) -c aux/reference.cu

rounder_test.o: rounder_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

interval_test.o: interval_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

box_test.o: box_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

cuda_box_test.o: cuda_box_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<


#serial blas
serial_blas1_test.o: serial_blas1_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

serial_blas2_test.o: serial_blas2_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

serial_blas3_test.o: serial_blas3_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<


#omp blas
omp_blas1_test.o: omp_blas1_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

omp_blas2_test.o: omp_blas2_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

omp_blas3_test.o: omp_blas3_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<


# CUDA portion
device_data_test.o: device_data_test.cu ../include
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

cuda_interval_test.o: cuda_interval_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

cuda_blas1_test.o: cuda_blas1_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

cuda_blas2_test.o: cuda_blas2_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<

cuda_blas3_test.o: cuda_blas3_test.cu ../include reference.o
	$(CC) $(PATHS) $(FLAGS) $(PTHREAD) -c $<



clean:
	rm test *.o

