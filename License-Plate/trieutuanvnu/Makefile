PROJECT = out
SRC = RecogSVM.cpp
LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	g++ $(SRC) -o $(PROJECT) $(LIBS)