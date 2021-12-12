#pragma once

#include "ModuleM.h"
#include "ModuleV.h"
#include "ModuleC.cuh"

#include <SDL2/SDL.h>

class Game
{
public:
	Game();
	Game(const std::vector<Point>& pointForBeginning);
	~Game();

	void runGame();
private:
	Data data;
	GameViewer viewer;
	GameTicker ticker;

	void initSdl();

	SDL_Window* wnd;
	SDL_Renderer* rend;
};

