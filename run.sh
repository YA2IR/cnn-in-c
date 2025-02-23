#! /bin/bash

clang -O3 -o main main.c matrix.c cnn.c && ./main

# uncomment and customize the following if OpenMP is desired
# clang -O3 -march=native -Wall -Xclang -fopenmp -fopenmp-simd  -framework Accelerate  -L/opt/homebrew/opt/libomp/lib  -I/opt/homebrew/opt/libomp/include  -lomp  -o main main.c matrix.c cnn.c && ./main


# debugging version w address sanitizer
# clang -g -O1 -fsanitize=address -Wall -o main main.c matrix.c cnn.c && ./main

