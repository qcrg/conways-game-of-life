#include "kernel.cuh"

#include <mutex>
#include <iostream>
#include <algorithm>
#include <ranges>

const Game::SyncUnit Game::getAliveCells() const
{
	alive_cells.syncCache();
	return { alive_cells.cache, std::lock_guard<std::mutex>(alive_cells.cache_mutex) };
}

void Game::Alive::syncCache() const
{
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	std::thread([&]()
		{
			std::scoped_lock guard(cache_mutex, data_mutex);
			if (data.size() > cache.capacity()) cache.reserve(data.size());
			cache.resize(0);
			std::copy(data.begin(), data.end(), std::back_inserter(cache));
		}).detach();
}

void Game::setCell(Coords cell)
{
	bool* tmp = game_field + COORD(cell.x, cell.y, max.y);
	std::scoped_lock guard(alive_cells.cache_mutex, alive_cells.data_mutex);
	if (*tmp)
	{
		alive_cells.data.erase(cell);
		alive_cells.cache.erase(std::ranges::find(alive_cells.cache, cell));
	}
	else
	{
		alive_cells.data.insert(cell);
		alive_cells.cache.push_back(cell);
	}
	*tmp = !*tmp;
}

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

void checkK(cudaError_t error, std::string message)
{
	if (error != cudaSuccess)
	{
		std::cerr << message << ": " << cudaGetErrorString(error) << "\n";
		throw - 1;
	}
}

void Game::tick()
{
	std::vector<Coords, HostAllocator<Coords>> check(alive_cells.data.begin(), alive_cells.data.end());

	for (auto& cell : alive_cells.data)
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


	checkCell(changed.data(), check.data(), check.size(), max, game_field, grid_size, block_size);


	std::scoped_lock<std::mutex> syn_gurds(alive_cells.data_mutex);

	for (unsigned int i = 0; i < changed.size(); ++i)
	{
		auto cell = changed[i];
		if (!(cell == Coords{ UINT_MAX, UINT_MAX }))
		{
			auto it = alive_cells.data.insert(cell);
			bool debug = game_field[COORD(cell.x, cell.y, max.y)];
			game_field[COORD(cell.x, cell.y, max.y)] = it.second;
			if (!it.second)
			{
				alive_cells.data.erase(it.first);
			}
		}
	}
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