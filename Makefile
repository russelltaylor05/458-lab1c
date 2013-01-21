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
	$(TARGET) input/A.in input/A.in
	@echo ""

test2: $(TARGET)
	$(TARGET) input/B.in input/B.in


clean: 
	rm -f *.o mm_cuda