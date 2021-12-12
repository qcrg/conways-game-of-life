#include "defines.h"

#include "ModuleV.h"

#include <cmath>

CoordinateConverter CC;

void setBlackCol(SDL_Renderer* r)
{
	sdlCheck(SDL_SetRenderDrawColor(r, 0, 0, 0, SDL_ALPHA_OPAQUE));
}

void setRedCol(SDL_Renderer* r)
{
	sdlCheck(SDL_SetRenderDrawColor(r, 255, 0, 0, SDL_ALPHA_OPAQUE));
}

void setTorquoiseCol(SDL_Renderer* r)
{
	sdlCheck(SDL_SetRenderDrawColor(r, 48, 213, 200, SDL_ALPHA_OPAQUE));
}

GameViewer::GameViewer(Data& ptr)
	: data{ptr}
	, screenPos{0, 0}
	, screenSize{DEFAULT_SCREEN_SIZE_W, DEFAULT_SCREEN_SIZE_H}
{}

void GameViewer::view(SDL_Renderer* render)
{
	setBlackCol(render);
	SDL_RenderClear(render);

	setRedCol(render);
	for (int64_t y = screenPos.get<1>(); y < screenSize.get<1>() + screenPos.get<1>(); y++)
	{
		for (int64_t x = screenPos.get<0>(); x < screenSize.get<0>() + screenPos.get<0>(); x++)
		{
			if (x >= 0
				&& x < data.field.getMaxSize().get<0>()
				&& y >= 0
				&& y < data.field.getMaxSize().get<1>())
			{
				Point tmpPoint(x, y);
				if (!data.field[tmpPoint]) continue;


				SDL_Rect rect = CC.gamePointToDisplayRect(tmpPoint);
				rect.x -= screenPos.get<0>();
				rect.y -= screenPos.get<1>();
				SDL_RenderFillRect(render, &rect);
			}
		}
	}

	SDL_RenderPresent(render);
}

void GameViewer::setScreenSize(Point newSize)
{
	screenSize = std::move(newSize);
}

const Point& GameViewer::getSreenSize() const
{
	return screenSize;
}

const Point& GameViewer::getScreenPos() const
{
	return screenPos;
}

void GameViewer::setScreenPos(Point newPos)
{
	screenPos = std::move(newPos);
}

void GameViewer::moveScreenPos(const Point& shift)
{
	double shiftX = shift.get<0>(); //TODO speed
	double shiftY = shift.get<1>();
	int64_t x = (int64_t)round(screenPos.get<0>() + shiftX);
	int64_t y = (int64_t)round(screenPos.get<1>() + shiftY);

	screenPos = { x, y };
}

std::pair<Point, bool> CoordinateConverter::displayToGame(const Point& p) const
{
	int x = static_cast<int>(round(p.get<0>() / scaleFactor));
	int y = static_cast<int>(round(p.get<1>() / scaleFactor));
	Point resPoint = { x, y };
	SDL_Rect rect = gamePointToDisplayRect(resPoint);

	bool resBool = rect.x <= p.get<0>()
		&& rect.y <= p.get<1>()
		&& rect.x + rect.w >= p.get<0>()
		&& rect.y + rect.h >= p.get<1>();

	return { resPoint, resBool };
}

Point CoordinateConverter::gameToDisplay(const Point& p) const
{
	int x = static_cast<int>(round(p.get<0>() * scaleFactor));
	int y = static_cast<int>(round(p.get<1>() * scaleFactor));
	return { x, y };
}

SDL_Rect CoordinateConverter::gamePointToDisplayRect(const Point& centerGamePoint) const
{
	Point point = CC.gameToDisplay(centerGamePoint);
	float foo = (scaleFactor / 2 - 1);
	int x = static_cast<int>(round(static_cast<float>(point.get<0>()) - foo));
	int y = static_cast<int>(round(static_cast<float>(point.get<1>()) - foo));
	int w = static_cast<int>(round(scaleFactor - 1.f));
	int h = w;
	return { x, y, w, h};
}