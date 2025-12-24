
## Getting Started

1. Clone the repo.
2. Create an external dir:
```
    mkdir external
```
4. Go to this website: https://pytorch.org/get-started/locally/.
5. Select your OS, Libtorch, C++/Java, and choose your build platform.
6. Unzip the files and put the libtorch folder into external.
7. In the root of the repo, perform these commands:
```
    mkdir build && cd build
    cmake ..
    make
    ./main
```

## Credit to
https://github.com/archishou/MidnightMoveGen for move generation.
https://github.com/pytorch/pytorch for model training and evaluation.
