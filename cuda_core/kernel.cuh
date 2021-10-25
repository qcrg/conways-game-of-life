#include <set>
#include "cuda_runtime.h"

#define COORD(x, y, max_y) (y * max_y + x)

struct Coords
{
	unsigned int x;
	unsigned int y;
};

bool operator== (const Coords& lhs, const Coords& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool operator< (const Coords& lhs, const Coords& rhs)
{
	bool c = lhs.x < rhs.x;
	if (c) return c;
	return lhs.x == rhs.x ? lhs.y < rhs.y : false;
}

class Game
{
public:
	Game(unsigned int max_x, unsigned int max_y);
	~Game();
	void tick();
	void setCell(Coords cell);
	const std::set<Coords>& getAliveCells() const;
private:
	bool* game_field;
	std::set<Coords> alive_cells;
	const Coords max;
	cudaDeviceProp current_device;
};