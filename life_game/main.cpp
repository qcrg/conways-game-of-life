#include <SDL.h>
#include <iostream>
#include <thread>
#include <condition_variable>
#include <chrono>


#include "game.h"
#include "input.h"
#include "const_values.h"

import Sdl;
import Point;

static int SCREEN_WIDTH = 640;
static int SCREEN_HEIGHT = 480;
static int OFFSET_X = 0;
static int OFFSET_Y = 0;
static pnd::Point2 SCALE = { 1, 1, 1 };
static pnd::Point2 ORIGIN = { 0, 0, 0 };

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
	bool one_beat = false;
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
				case SDL_MOUSEMOTION:
					{
						if(mEvent.motion.state & SDL_BUTTON_MIDDLE)
						{
							OFFSET_X += mEvent.motion.xrel;
							OFFSET_Y += mEvent.motion.yrel;
						}
					}
					break;
				case SDL_MOUSEWHEEL:
					{
						float scale = (mEvent.wheel.y > 0 ? 2.0f : 0.5f);
						SCALE = pnd::scale2(&ORIGIN, &SCALE, scale);
					}
					break;
				case SDL_MOUSEBUTTONUP:
					{
						if (mEvent.button.button & SDL_BUTTON_LEFT)
						{
							int foo_x = (mEvent.button.x / SCALE.x - OFFSET_X) / 10.0f;
							int foo_y = (mEvent.button.y / SCALE.y - OFFSET_Y) / 10.0f;
							gm.setCell(foo_x, foo_y);
						}
					}
				case SDL_KEYDOWN:
					{
						switch(mEvent.key.keysym.sym)
						{
							case SDLK_r:
								{
									play = !play;
								}
								break;
							case SDLK_d:
								{
									one_beat = true;
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

		
		SDL_Delay(33);

		if (play)
		{
			gm._one_beat();
		}
		else if (one_beat)
		{
			gm._one_beat();
			one_beat = false;
		}
		
		draw(sdl.getRenderer().get(), gm);

	}
	

	return 0;
}

void drawRect(SDL_Renderer* rend, int x, int y)
{
	int rect_size = 8 * SCALE.x;
	SDL_Rect rect{ (OFFSET_X + 10 * x) * SCALE.x, (OFFSET_Y + 10 * y) * SCALE.y, rect_size, rect_size};
	SDL_RenderFillRect(rend, &rect);
}

void draw(SDL_Renderer* rend, game& gm)
{
	SDL_SetRenderDrawColor(rend, 0, 0, 0, 255);
	SDL_RenderClear(rend);

	SDL_SetRenderDrawColor(rend, 48, 213, 200, 255);

	SDL_Rect rect{ (OFFSET_X - 2) * SCALE.x, (OFFSET_Y - 2) * SCALE.y, MAX_GAME_FIELD_X * SCALE.x * 10 + 2, MAX_GAME_FIELD_Y * SCALE.y * 10 + 2};
	SDL_RenderDrawRect(rend, &rect);

	SDL_SetRenderDrawColor(rend, 255, 0, 0, 255);

	int x_ = 0;
	int y_ = 0;
	for (const auto& __x : gm._show_game_field())
	{
		for (const auto& __y : __x)
		{
			if(__y) drawRect(rend, x_, y_);
			++y_;
		}
		y_ = 0;
		++x_;
	}

	SDL_RenderPresent(rend);
}