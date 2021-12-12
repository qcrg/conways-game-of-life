#pragma once

#include "ModuleM.h"

#include <SDL2/SDL.h>

class GameViewer
{
public:
	GameViewer(Data& ptr);

	void view(SDL_Renderer*);
	void setScreenSize(Point newSize);
	const Point& getSreenSize() const;

	void setScreenPos(Point newPos);
	void moveScreenPos(const Point& shift);
	const Point& getScreenPos() const;
private:
	Data& data;
	Point screenPos;
	Point screenSize;
};

class CoordinateConverter
{
public:
	CoordinateConverter(float factor = 10.f) : scaleFactor{ factor } {}
	std::pair<Point, bool> displayToGame(const Point& p) const;
	Point gameToDisplay(const Point& p) const;
	SDL_Rect gamePointToDisplayRect(const Point& centerGamePoint) const;

	const float scaleFactor;
};