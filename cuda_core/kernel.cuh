#include <set>
#include "cuda_runtime.h"

#define COORD(x, y, max_y) (y * max_y + x)

struct Coords
{
	unsigned int x;
	unsigned int y;
};

bool operator== (const Coords& lhs, const Coords& rhs);

bool operator< (const Coords& lhs, const Coords& rhs);

class Game
{
public:
	Game(unsigned int max_x, unsigned int max_y);
	~Game();
	void tick();
	void setCell(Coords cell);
	const std::set<Coords>& getAliveCells() const;

	const Coords max;
private:
	bool* game_field;
	std::set<Coords> alive_cells;
	cudaDeviceProp current_device;
};