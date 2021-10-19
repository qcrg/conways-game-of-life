export module Sdl;

import <cstdint>;
import <SDL.h>;
import <memory>;
import <string>;


export struct SdlDeleterForWindow
{
	void operator()(SDL_Window* pointer)
	{
		SDL_DestroyWindow(pointer);
	}
};
export struct SdlDeleterForSurface
{
	void operator()(SDL_Surface* pointer)
	{
		SDL_FreeSurface(pointer);
	}
};
export struct SdlDeleterForRenderer
{
	void operator()(SDL_Renderer* pointer)
	{
		SDL_DestroyRenderer(pointer);
	}
};


export class Sdl
{
public:
	Sdl(uint32_t flags_for_init_sdl);
	~Sdl();
public:
	void initWindow(std::string title, int x, int y, int w, int h, uint32_t flags);
	void initSurface();
	void initRenderer(int index_driver, uint32_t flags);
public:
	std::unique_ptr<SDL_Window, SdlDeleterForWindow>& getWindow();
	std::unique_ptr<SDL_Surface, SdlDeleterForSurface>& getSurface();
	std::unique_ptr<SDL_Renderer, SdlDeleterForRenderer>& getRenderer();

private:
	std::unique_ptr<SDL_Window, SdlDeleterForWindow> window;
	std::unique_ptr<SDL_Surface, SdlDeleterForSurface> surface;
	std::unique_ptr<SDL_Renderer, SdlDeleterForRenderer> renderer;
};





















void printErrorAndQuit(bool if_true_then_quit, int error_code)
{
	if (if_true_then_quit)
	{
		SDL_LogError(error_code, "%s", SDL_GetError());
		exit(error_code);
	}
}

Sdl::Sdl(uint32_t flags)
{
	printErrorAndQuit(SDL_Init(flags) != 0, -1);
}

Sdl::~Sdl()
{
	renderer.reset();
	surface.reset();
	window.reset();
	SDL_Quit();
}

void Sdl::initWindow(std::string title, int x, int y, int w, int h, uint32_t flags)
{
	window.reset(SDL_CreateWindow(title.data(), x, y, w, h, flags));
}

void Sdl::initSurface()
{
	surface.reset(SDL_GetWindowSurface(window.get()));
}

void Sdl::initRenderer(int index_driver, uint32_t flags)
{
	renderer.reset(SDL_CreateRenderer(window.get(), index_driver, flags));
}

std::unique_ptr<SDL_Window, SdlDeleterForWindow>& Sdl::getWindow()
{
	return window;
}

std::unique_ptr<SDL_Surface, SdlDeleterForSurface>& Sdl::getSurface()
{
	return surface;
}

std::unique_ptr<SDL_Renderer, SdlDeleterForRenderer>& Sdl::getRenderer()
{
	return renderer;
}
