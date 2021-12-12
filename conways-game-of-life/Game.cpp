#include "defines.h"

#include "Game.h"


Game::Game()
	: viewer{ data }
	, ticker{ data }
{
	initSdl();

	data.changeCell({ 4, 3 });
	data.changeCell({ 4, 4 });
	data.changeCell({ 4, 5 });
}

Game::Game(const std::vector<Point>& pointForBeginning)
	: viewer{ data }
	, ticker{ data }
{
	for (const auto& point : pointForBeginning)
	{
		data.changeCell(point);
	}
	initSdl();
}

Game::~Game()
{
	SDL_DestroyWindow(wnd);
	SDL_Quit();
}

void Game::runGame()
{
	bool quit = false;
	bool oneTick = false;
	bool play = false;
	CoordinateConverter cConverter;
	while (!quit)
	{
		SDL_Event event;
		while (SDL_PollEvent(&event) > 0)
		{
			switch (event.type)
			{
			case SDL_WINDOWEVENT:
			{
				switch (event.window.event)
				{
				case SDL_WINDOWEVENT_RESIZED:
				{
					viewer.setScreenSize({event.window.data1, event.window.data2});
				}
				break;
				}
			}
			break;
			case SDL_MOUSEMOTION:
			{
				if (event.motion.state & SDL_BUTTON_MIDDLE)
				{
					viewer.moveScreenPos({event.motion.xrel, event.motion.yrel});
				}
			}
			break;
			case SDL_MOUSEWHEEL:
			{
				//TODO scale
			}
			break;
			case SDL_MOUSEBUTTONUP:
			{
				if (event.button.button & SDL_BUTTON_LEFT)
				{
					std::pair<Point, bool> tmp = cConverter.displayToGame({ event.button.x, event.button.y });
					if (tmp.second)
					{
						data.changeCell(tmp.first);
					}
				}
			}
			break;
			case SDL_KEYDOWN:
			{
				switch (event.key.keysym.sym)
				{
				case SDLK_r:
				{
					play = !play;
				}
				break;
				case SDLK_d:
				{
					oneTick = true;
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

		if (play)
		{
			ticker.runOneStep();
		}
		else if (oneTick)
		{
			ticker.runOneStep();
			oneTick = false;
		}

		viewer.view(rend);
	}
}

void Game::initSdl()
{
	sdlCheck(SDL_Init(SDL_INIT_EVENTS | SDL_INIT_VIDEO));
	wnd = SDL_CreateWindow(
		"Jon Conway's Game of Life"
		, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED
		, DEFAULT_SCREEN_SIZE_W, DEFAULT_SCREEN_SIZE_H
		, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

	rend = SDL_CreateRenderer(wnd, -1, SDL_RENDERER_ACCELERATED);

	assert(wnd != nullptr);
	assert(rend != nullptr);
}