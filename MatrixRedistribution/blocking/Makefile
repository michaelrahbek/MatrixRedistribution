OBJECTS   = matrix_functions.o index_size_functions.o redistribute.o warm_up.o
OBJECTS_R = matrix_functions.o index_size_functions.o redistribute_test.o warm_up.o

CC		= mpicc
CFLAGS	= -Wall -std=c99 -O3 -march=native
LDFLAGS = -lm 

all: project redistribute_test.o test

project: $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJECTS) main.c

redistribute_test.o:
	$(CC) $(CFLAGS) -D TEST -c -o $@ redistribute.c

test: $(OBJECTS_R)
	$(CC) $(CFLAGS) -D TEST $(LDFLAGS) -o $@ $(OBJECTS_R) main.c

.PHONY: clean
clean:
	-$(RM) project test *.o