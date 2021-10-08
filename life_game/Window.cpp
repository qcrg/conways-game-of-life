#include "Window.h"
#include "const_values.h"
#include "game.h"
#include "input.h"

#include <iostream>

SDL_DisplayMode dm;


ApplicationInfos appInfos = {
	"Conway's game of life",
	640,
	480
};



Window::Window()
{
	if (!init())
	{
		std::cerr << "Failled initialize.\n";
	}
}

Window::~Window()
{
	SDL_DestroyWindow(hWnd);
	hWnd = NULL;
	SDL_Quit();
}

bool Window::init()
{
	bool success = true;

	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		std::cerr << "Failded SDL init. SDL Error: " << SDL_GetError() << "\n";
		success = false;
	}
	else
	{
		hWnd = SDL_CreateWindow(appInfos.appName, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, appInfos.defaultScreenWigth, appInfos.defaultScreenHeight, SDL_WINDOW_SHOWN);
		if (hWnd == NULL)
		{
			std::cerr << "Window could not be created. SDL_Error: " << SDL_GetError() << "\n";
			success = false;
		}
		else
		{
			hScreenSurface = SDL_GetWindowSurface(hWnd);
		}
	}

	return success;
}

void Window::mainLoop()
{
	bool quit = false;

	SDL_Event hEvent;

	input hInput("game_info");

	game hGame(hInput.get_alive_cels());

	while (!quit)
	{
		SDL_Delay(16);
		
		while (SDL_PollEvent(&hEvent) != 0)
		{
			switch (hEvent.type)
			{
				
			case SDL_QUIT:
			{
				quit = true;
			}
			}
		}
		
		hGame._one_beat();

	}
}