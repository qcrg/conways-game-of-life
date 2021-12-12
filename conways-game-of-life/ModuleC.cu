#include "defines.h"

#include "basic_types.h"
#include "ModuleC.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

GameTicker::GameTicker(Data& ptr)
	: data{ptr}
{
	cuCHECK(cudaGetDeviceProperties(&currentDevice, 0));
}

template <typename T>
struct HostAllocator
{
	typedef T value_type;
	inline T* allocate(const size_t count)
	{
		T* tmp;
		cuCHECK(cudaMallocHost(&tmp, sizeof(T) * count));
		return tmp;
	}
	inline void deallocate(T* const ptr, const size_t count)
	{
		cuCHECK(cudaFreeHost(ptr));
	}
};

__host__ __device__ void set_range_change(int64_t& c_min, int64_t& c_max, int64_t coord, int64_t max_value)
{
	if (coord == 0) {
		c_min = 0;
		c_max = 1;
	}
	else if (coord > 0 && coord != max_value - 1) {
		c_min = -1;
		c_max = 1;
	}
	else {//when cell_coord == max_value - 1
		c_min = -1;
		c_max = 0;
	}
}

__global__ void checkCellCuda(Point* changed, Point* check, int64_t check_size, Point max, bool* game_field)
{
	int64_t y_min, y_max;
	int64_t x_min, x_max;

	int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_idx >= check_size) return;

	Point cell = check[global_idx];

	set_range_change(x_min, x_max, cell.get<0>(), max.get<0>());
	set_range_change(y_min, y_max, cell.get<1>(), max.get<1>());

	int alive_cells = 0;
	for (int tmp_y = y_min; tmp_y <= y_max; ++tmp_y)
	{
		for (int tmp_x = x_min; tmp_x <= x_max; ++tmp_x)
		{
			unsigned int x = cell.get<0>() + tmp_x;
			unsigned int y = cell.get<1>() + tmp_y;
			bool* cell_value = game_field + max.get<1>() * y + x;

			if (*cell_value)
				++alive_cells;
		}
	}

	bool* cell_in_game_field = game_field + max.get<1>() * cell.get<1>() + cell.get<0>();
	bool have_3_or_4_alive_cells = (3 == alive_cells || alive_cells == 4);
	bool change_cage = *cell_in_game_field ^ (*cell_in_game_field ? have_3_or_4_alive_cells : 3 == alive_cells);

	if (change_cage)
	{
		changed[global_idx] = cell;
	}
	else
	{
		changed[global_idx] = Point{ -1, -1 };
	}
}

#if defined(__INTELLISENSE__)
#define KERNEL_ARGS2(grid, block)
#define __CUDACC__
#else
#define KERNEL_ARGS2(grid, block) <<<grid, block>>>
#endif

void GameTicker::runOneStep()
{
	std::vector<Point, HostAllocator<Point>> check(data.aliveCells.begin(), data.aliveCells.end());

	for (auto& cell : data.aliveCells) // optimize with fibers
	{
		int64_t x_min, x_max;
		int64_t y_min, y_max;

		set_range_change(x_min, x_max, cell.get<0>(), data.field.getMaxSize().get<0>());
		set_range_change(y_min, y_max, cell.get<1>(), data.field.getMaxSize().get<1>());
		for (int64_t tmp_y = y_min; tmp_y <= y_max; ++tmp_y)
		{
			for (int64_t tmp_x = x_min; tmp_x <= x_max; ++tmp_x)
			{
				int64_t x = cell.get<0>() + tmp_x;
				int64_t y = cell.get<1>() + tmp_y;

				Point tmp_cell{ x, y };
				auto it = std::lower_bound(check.begin(), check.end(), tmp_cell, CrutchPredLess{});

				if (it == check.end() || !(*it == tmp_cell))
				{
					check.emplace(it, std::move(tmp_cell));
				}
			}
		}
	}

	std::vector<Point, HostAllocator<Point>> changed;
	changed.resize(check.size());

	int size = (int)check.size(), th = currentDevice.maxThreadsPerBlock;
	dim3 block_size(size > th ? th : size);
	dim3 grid_size(size / th + 1);
	
	checkCellCuda KERNEL_ARGS2(grid_size, block_size)(
		changed.data()
		, check.data()
		, check.size()
		, data.field.getMaxSize()
		, data.field.data());

	cuCHECK(cudaDeviceSynchronize());

	for (auto& point : changed) //optimize with fibers
	{
		if (!(point == Point{ -1, -1 }))
			data.changeCell(std::move(point));
	}
}
