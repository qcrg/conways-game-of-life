
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <vector>
#include <set>
#include <thread>
#include <algorithm>
#include <iostream>

#include "kernel.cuh"

#if defined(__INTELLISENSE__)
#define KERNEL_ARGS2(grid, block)
#define __CUDACC__
#else
#define KERNEL_ARGS2(grid, block) <<<grid, block>>>
#endif

#define SCOPE(foo) #foo
#define PROXY_SCOPE(bar) SCOPE(bar)
#define ERROR_MSG (__FILE__ "\t line " PROXY_SCOPE(__LINE__))
#define CHECK_K(func) checkK(func, ERROR_MSG)


__host__ __device__ void set_range_change(int* c_min, int* c_max, const unsigned int coord, const unsigned int max_value)
{
	if (coord == 0) {
		*c_min = 0;
		*c_max = 1;
	}
	else if (coord > 0 && coord != max_value - 1) {
		*c_min = -1;
		*c_max = 1;
	}
	else {//when cell_coord == max_value - 1
		*c_min = -1;
		*c_max = 0;
	}
}

void checkK(cudaError_t error, std::string message)
{
	if (error != cudaSuccess)
	{
		std::cout << message << ": " << cudaGetErrorString(error) << "\n";
		throw - 1;
	}
}

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



Game::Game(unsigned int max_x, unsigned int max_y)
	: max({ max_x, max_y })
{
	CHECK_K(cudaGetDeviceProperties(&current_device, 0));
	CHECK_K(cudaMallocHost(&game_field, sizeof(bool) * max_x * max_y));
}

Game::~Game()
{
	CHECK_K(cudaFreeHost(game_field));
}

__global__ void ckeckCell(Coords* changed, Coords* check, unsigned int check_size, Coords max, bool* game_field)
{
	int y_min, y_max;
	int x_min, x_max;

	//printf("gridDim.x: %d\nblockIdx.x: %d\nthreadIdx.x: %d\n\n\n", gridDim.x, blockIdx.x, threadIdx.x);
	unsigned int global_idx = gridDim.x * blockIdx.x * blockDim.x + threadIdx.x;

	if (global_idx >= check_size) return;

	Coords cell = check[global_idx];
	//printf("(%d, %d)\n", cell.x, cell.y);

	set_range_change(&y_min, &y_max, cell.y, max.y);
	set_range_change(&x_min, &x_max, cell.x, max.x);

	int alive_cells = 0;
	for (int tmp_y = y_min; tmp_y <= y_max; ++tmp_y)
	{
		for (int tmp_x = x_min; tmp_x <= x_max; ++tmp_x)
		{
			unsigned int x = cell.x + tmp_x;
			unsigned int y = cell.y + tmp_y;
			bool* cell_value = game_field + COORD(x, y, max.y);
			//if(global_idx == 4)printf("global_idx(%u), (%u, %u): %s\n",global_idx, x, y, *cell_value ? "true" : "false");//del

			if (*cell_value)
				++alive_cells;
		}
	}

	//printf("global_idx: %d\t Alive cells: %d\t (%d, %d)\n", global_idx, alive_cells, cell.x, cell.y);//del

	
	bool *cell_in_game_field = game_field + COORD(cell.x, cell.y, max.y);
	bool have_3_or_4_alive_cells = (3 == alive_cells || alive_cells == 4);
	bool change_cage = *cell_in_game_field ^ have_3_or_4_alive_cells;

	if (change_cage)
	{
		//printf("global_idx: %d; (%u, %u); cell_in_game_field: %d; have_3_or_4_alive_cells: %d; change_cage: %d\n",
		//	global_idx, cell.x, cell.y, *cell_in_game_field, have_3_or_4_alive_cells, change_cage);//del

		changed[global_idx] = cell;
	}
	else
	{
		changed[global_idx] = Coords{ UINT_MAX, UINT_MAX };
	}
}

void Game::tick()
{
	std::vector<Coords, HostAllocator<Coords>> check(alive_cells.begin(), alive_cells.end());



	for (auto& cell : alive_cells)
	{
		int y_min, y_max;
		int x_min, x_max;

		set_range_change(&y_min, &y_max, cell.y, max.y);
		set_range_change(&x_min, &x_max, cell.x, max.x);
		for (int tmp_y = y_min; tmp_y <= y_max; ++tmp_y)
		{
			for (int tmp_x = x_min; tmp_x <= x_max; ++tmp_x)
			{
				unsigned int x = cell.x + tmp_x;
				unsigned int y = cell.y + tmp_y;

				Coords tmp_cell{ x, y };
				auto it = std::lower_bound(check.begin(), check.end(), tmp_cell);
				 
				if (!(*it == tmp_cell))
				{
					check.emplace(it, std::move(tmp_cell));
				}
			}
		}
	}

	std::vector<Coords, HostAllocator<Coords>> changed;
	changed.resize(check.size());

	unsigned int size = check.size(), th = current_device.maxThreadsPerBlock;
	dim3 block_size(size > th ? th : size);
	dim3 grid_size(size / th + 1);

	cudaEvent_t sync;
	CHECK_K(cudaEventCreate(&sync));
	CHECK_K(cudaEventRecord(sync, 0));
	ckeckCell KERNEL_ARGS2(grid_size, block_size) (changed.data(), check.data(), check.size(), max, game_field);
	CHECK_K(cudaEventSynchronize(sync));




	for (unsigned i = 0; i < changed.size(); ++i)
	{
		auto cell = changed[i];
		if (!(cell == Coords{ UINT_MAX, UINT_MAX }))
		{
			auto it = alive_cells.insert(cell);
			bool debug = game_field[COORD(cell.x, cell.y, max.y)];
			game_field[COORD(cell.x, cell.y, max.y)] = it.second;
			if (!it.second)
			{
				alive_cells.erase(it.first);
			}
		}
	}
	//CHECK_K(cudaEventDestroy(sync));
}

void Game::setCell(Coords cell)
{
	bool* tmp = game_field + COORD(cell.x, cell.y, max.y);
	if (*tmp)
	{
		alive_cells.erase(cell);
	}
	else
	{
		alive_cells.insert(cell);
	}
	*tmp = !*tmp;
}

const std::set<Coords>& Game::getAliveCells() const
{
	return alive_cells;
}



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