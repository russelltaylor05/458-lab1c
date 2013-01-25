NVFLAGS=-g -O2 -arch=compute_20 -code=sm_20 
#NVFLAGS=-g --ptxas-options="-v"
# list .c and .cu source files here
# use -02 for optimization during timed runs


SRCFILES=main.cu 
TARGET = ./mm_cuda

all:	mm_cuda	

mm_cuda: $(SRCFILES) 
	nvcc $(NVFLAGS) -o mm_cuda $^

double: $(SRCFILES)
	nvcc $(NVFLAGS) -DDOUBLE -o mm_cuda main.cu  

single: $(SRCFILES)
	nvcc $(NVFLAGS) -DSINGLE -o mm_cuda main.cu


test1: $(TARGET)
	$(TARGET) input/A.in input/B.in

test2: $(TARGET)
	$(TARGET) input/1408.in input/1408.in > result2.out
	./lupo_cuda input/1408.in input/1408.in
	diff result.out result2.out
	rm result.out result2.out

test3: $(TARGET)
	$(TARGET) input/555x666.in input/666x777.in > result2.out
	./lupo_cuda input/555x666.in input/666x777.in
	

clean: 
	rm -f *.o mm_cuda