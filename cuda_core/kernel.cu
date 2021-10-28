
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

__global__ void checkCellCuda(Coords* changed, Coords* check, unsigned int check_size, Coords max, bool* game_field)
{
	int y_min, y_max;
	int x_min, x_max;

	//printf("gridDim.x: %d\nblockIdx.x: %d\nthreadIdx.x: %d\n\n\n", gridDim.x, blockIdx.x, threadIdx.x);
	unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("check_size: %u\t\tglobal_idx: %u\n", check_size, global_idx);

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


	bool* cell_in_game_field = game_field + COORD(cell.x, cell.y, max.y);
	bool have_3_or_4_alive_cells = (3 == alive_cells || alive_cells == 4);
	bool change_cage = *cell_in_game_field ^ (*cell_in_game_field ? have_3_or_4_alive_cells : 3 == alive_cells);

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

void checkCell(Coords* changed, Coords* check, unsigned int check_size, Coords max, bool* game_field, dim3 grid_size, dim3 block_size)
{
	checkCellCuda KERNEL_ARGS2(grid_size, block_size) (changed, check, check_size, max, game_field);
	cudaDeviceSynchronize();
}