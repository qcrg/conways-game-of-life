#pragma once
#include <SDL.h>

class Window
{
public:
	Window();
	~Window();
public:
	void mainLoop();
private:
	SDL_Window* hWnd;
	SDL_Surface* hScreenSurface;
private:
	bool init();
};

struct ApplicationInfos
{
	const char appName[256];
	const int defaultScreenWigth;
	const int defaultScreenHeight;
};