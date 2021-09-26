CC = nvcc
OBJS = hw1.o user_device.o user_host.o user.o
CFLAGS =

all : hw1.out

hw1.out : $(OBJS)
	$(CC) -o hw1.out $(OBJS)

hw1.o : hw1.cu
	$(CC) -c $(CFLAGS) hw1.cu -o hw1.o

user_device.o : user_device.cu
	$(CC) -c $(CFLAGS) user_device.cu -o user_device.o

user_host.o : user_host.cu
	$(CC) -c $(CFLAGS) user_host.cu -o user_host.o

user.o : user.cu
	$(CC) -c $(CFLAGS) user.cu -o user.o

clean :
	rm -f hw1.out *.o

rebuild : clean all