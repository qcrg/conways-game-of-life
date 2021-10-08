#include <iostream>
#include <thread>
#include <chrono>

#include "Window.h"


int main(int argc, char* argv[]) {
	Window wnd;
	wnd.mainLoop();

	return 0;
}