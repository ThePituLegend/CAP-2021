all: neuralBase.x neuralPar.x

common.o: common.c
	gcc -c -Ofast -std=c99 -Wall common.c

nn-mainBase.o: nn-mainBase.c
	gcc -c -Ofast -std=c99 -Wall nn-mainBase.c

nn-mainPar.o: nn-mainPar.c
	gcc -c -Ofast -fopenmp -std=c99 -Wall nn-mainPar.c

neuralBase.x: common.o nn-mainBase.o
	gcc -Ofast common.o nn-mainBase.o -lm -o neuralBase.x

neuralPar.x: common.o nn-mainPar.o
	gcc -Ofast -fopenmp common.o nn-mainPar.o -lm -o neuralPar.x

clean:
	rm -rf *.o *.x