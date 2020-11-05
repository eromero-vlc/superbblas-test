all: group_ip

group_ip: group_ip.cu clean
	nvcc -o group_ip -Xcompiler '-Wall -Wextra -fopenmp' --std=c++11 -lopenblas -g -G group_ip.cu 

clean:
	rm -f group_ip
