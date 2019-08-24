# Compiler
COMPILER = g++

# Flags
# FLAGS = -pedantic -ansi -Wall -Wextra -Wundef -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -Wconversion -Wredundant-decls -Winline -Wstrict-null-sentinel -Wold-style-cast -Woverloaded-virtual -Wsign-promo -Wimplicit -Wformat=2 -Winit-self -Wmissing-include-dirs -Wstrict-aliasing=2  -Werror -Wfatal-errors
FLAGS = -ansi -lm

# Input Files
INPUT = CrossValidation.cpp File.cpp Forget.cpp Kernel.cpp OnlineSVR.cpp Show.cpp Stabilize.cpp Train.cpp Variations.cpp

all: executable1 executable2

# Compile Main
MAIN1 = Main.cpp
OUTPUT1 = OnlineSVR
executable1: 
	$(COMPILER) -o $(OUTPUT1) $(MAIN1) $(INPUT) $(FLAGS)

# Compile Example
MAIN2 = Example.cpp
OUTPUT2 = Example
executable2: 
	$(COMPILER) -o $(OUTPUT2) $(MAIN2) $(INPUT) $(FLAGS)
