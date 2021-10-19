#include <SDL.h>
#include <iostream>
#include <thread>
#include <condition_variable>
#include <chrono>

#include "game.h"
#include "input.h"

import Sdl;

static int SCREEN_WIDTH = 640;
static int SCREEN_HEIGHT = 480;

void draw(SDL_Renderer* rend, game& gm);

int main(int argc, char* argv[])
{
	Sdl sdl(SDL_INIT_VIDEO);
	sdl.initWindow("Conway's game of life", SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT,
		SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
	sdl.initRenderer(-1, SDL_RENDERER_ACCELERATED);

	input in("game_info");
	game gm(in.get_alive_cels());

	std::condition_variable cv;

	bool play = false;
	bool quit = false;
	while (!quit)
	{
				


		SDL_Event mEvent;
		while (SDL_PollEvent(&mEvent) > 0)
		{
			switch (mEvent.type)
			{
				case SDL_WINDOWEVENT:
					{
						switch(mEvent.window.event)
						{
							case SDL_WINDOWEVENT_RESIZED:
								{
									SDL_WindowEvent& __tmp = mEvent.window;
									SCREEN_WIDTH = __tmp.data1;
									SCREEN_HEIGHT = __tmp.data2;
								}
								break;
						}
					}
					break;
				case SDL_KEYDOWN:
					{
						switch(mEvent.key.keysym.sym)
						{
							case SDLK_p:
								{
									play = !play;
								}
								break;
						}
					}
					break;
				case SDL_QUIT:
					{
						quit = true;
					}
					break;
			}
		}

		

		draw(sdl.getRenderer().get(), gm);

	}
	

	return 0;
}



void draw(SDL_Renderer* rend, game& gm)
{

}