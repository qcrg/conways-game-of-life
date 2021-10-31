#include <concurrent_unordered_set.h>
#include "cuda_runtime.h"

#define COORD(x, y, max_y) (y * max_y + x)

#define SCOPE(foo) #foo
#define PROXY_SCOPE(bar) SCOPE(bar)
#define ERROR_MSG (__FILE__ "\t line " PROXY_SCOPE(__LINE__))
#define CHECK_K(func) checkK(func, ERROR_MSG)

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
	const concurrency::concurrent_unordered_set<Coords>& getAliveCells() const;

	const Coords max;
private:
	bool* game_field;
	concurrency::concurrent_unordered_set<Coords> alive_cells;
	cudaDeviceProp current_device;
};



void checkK(cudaError_t error, std::string message);
void checkCell(Coords* changed, Coords* check, unsigned int check_size, Coords max, bool* game_field, dim3 grid_size, dim3 block_size);
__host__ __device__ void set_range_change(int* c_min, int* c_max, const unsigned int coord, const unsigned int max_value);



template<typename T>
struct HostAllocator
{
	typedef T value_type;
	T* allocate(size_t count)
	{
		T* tmp;
		CHECK_K(cudaMallocHost(&tmp, count * sizeof(T)));
		return tmp;
	}
	void deallocate(T* ptr, size_t count)
	{
		CHECK_K(cudaFreeHost(ptr));
	}
};