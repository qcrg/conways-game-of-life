#pragma once

#include <vector>
#include <set>

#include "cell.h"


	class game
	{
	public:
		game(const std::set<std::pair<coord_t, coord_t>>& coords_alive_cels);

		const std::vector<std::vector<bool>>& _show_game_field() const;

		void _one_beat();

		void setCell(coord_t x, coord_t y);

		const std::set<cell_coord>& getAliveCells() const;

	private:
																					//		 xxx
		std::vector<std::vector<bool>> _game_field; // x * MAX_GAME_FIELD_Y + y		//		y000
		std::set<cell_coord> _alive_cells;											//		y000
	};																				//		y000
