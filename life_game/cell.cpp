#include "cell.h"
#include <utility>

cell::cell(const short y, const short x, const bool alive) 
	: _y_coord(y),
	_x_coord(x),
	_alive(alive)
{}

const bool cell::_u_alive() const
{
	return _alive;
}

void cell::_kill()
{
	_alive = false;
}

void cell::_resurrect()
{
	_alive = true;
}

const coord_t cell::_show_y_coord() const
{
	return _y_coord;
}

const coord_t cell::_show_x_coord() const
{
	return _x_coord;
}