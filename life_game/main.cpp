#include <SDL.h>
#include <iostream>
#include <thread>
#include <condition_variable>
#include <chrono>

#include "../cuda_core/kernel.cuh"

import Sdl;
import Point;

int SCREEN_WIDTH = 640;
int SCREEN_HEIGHT = 480;

float OFFSET_X = 0;
float OFFSET_Y = 0;

float SHEAR_SIZE = 5.0f;
float RECT_SIZE = SHEAR_SIZE * 0.8f;

pnd::Point2 SCALE = { 1, 1, 1 };
pnd::Point2 ORIGIN = { 0, 0, 0 };

void draw(SDL_Renderer* rend, Game& gm);
#include "main_funcs.h"

int main(int argc, char* argv[])
{
	Sdl sdl(SDL_INIT_VIDEO);
	sdl.initWindow("Conway's game of life", SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT,
		SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
	sdl.initRenderer(-1, SDL_RENDERER_ACCELERATED);

	Game gm(1000, 1000);
	gm.setCell({ 0, 0 });
	gm.setCell({ 0, 1 });
	gm.setCell({ 1, 0 });
	gm.setCell({ 2, 0 });

	bool play = true;
	bool one_beat = false;
	int speed = 33;
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
									windowResized(SCREEN_WIDTH, SCREEN_HEIGHT, mEvent.window);
								}
								break;
						}
					}
					break;
				case SDL_MOUSEMOTION:
					{
						mouseMotionEvent(OFFSET_X, OFFSET_Y, mEvent.motion);
					}
					break;
				case SDL_MOUSEWHEEL:
					{
						scaleWithWheel(ORIGIN, SCALE, mEvent.wheel);
					}
					break;
				case SDL_MOUSEBUTTONUP:
					{
						killCell(gm, mEvent.button);
					}
					break;
				case SDL_KEYDOWN:
					{
						keyDown(play, one_beat, speed, mEvent.key);
					}
					break;
				case SDL_QUIT:
					{
						quit = true;
					}
					break;
			}
		}


		SDL_Delay(speed);
		oneBeat(gm, one_beat, play);


		draw(sdl.getRenderer().get(), gm);

	}
	

	return 0;
}

void drawRect(SDL_Renderer* rend, int x, int y)
{
	int rect_size = RECT_SIZE * SCALE.x;
	SDL_Rect rect{ (OFFSET_X + SHEAR_SIZE * x) * SCALE.x, (OFFSET_Y + SHEAR_SIZE * y) * SCALE.y, rect_size, rect_size};
	SDL_RenderFillRect(rend, &rect);
}

void draw(SDL_Renderer* rend, Game& gm)
{
	SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
	SDL_RenderClear(rend);

	SDL_SetRenderDrawColor(rend, 48, 213, 200, 255);

	//for(int i = 0;)

	SDL_Rect rect{ (OFFSET_X - 2) * SCALE.x, (OFFSET_Y - 2) * SCALE.y, gm.max.x * SCALE.x * SHEAR_SIZE + 2, gm.max.y * SCALE.y * SHEAR_SIZE + 2};
	SDL_RenderDrawRect(rend, &rect);

	SDL_SetRenderDrawColor(rend, 255, 0, 0, 255);


	for (const auto& _cell : gm.getAliveCells())
	{
		drawRect(rend, _cell.x, _cell.y);
	}
	SDL_RenderPresent(rend);
}

