#include <algorithm>

#include "game.h"
#include "const_values.h"

//should always be signed
typedef int tmp_coord_t;

void set_range_change(const coord_t&, tmp_coord_t&, tmp_coord_t&, const size_t&);
void set_change_list(std::set<cell_coord>& list, std::vector<cell_coord>& _to_change, std::vector<std::vector<bool>>& _game_field, const bool _add_died_cells_check, std::set<cell_coord>& _died_cells_check);

game::game(const std::set<std::pair<coord_t, coord_t>>& coords_alive_cells)
{
	_game_field.resize(MAX_GAME_FIELD_Y);
	for (coord_t y = 0; y < MAX_GAME_FIELD_Y; ++y) {
		for (coord_t x = 0; x < MAX_GAME_FIELD_X; ++x) {
			_game_field[y].push_back(false);
		}
	}

	for (const auto& it : coords_alive_cells) {
		coord_t
			y = it.first,
			x = it.second;
		_alive_cells.insert({ y, x });
		_game_field[y][x] = true;
	}
}

const std::vector<std::vector<bool>>& game::_show_game_field() const
{
	return _game_field;
}

void game::_one_beat()
{
	std::vector<cell_coord> changed;
	std::set<cell_coord> _died_cells_check;

	set_change_list(_alive_cells, changed, _game_field, true, _died_cells_check);
	set_change_list(_died_cells_check, changed, _game_field, false, _died_cells_check);




	for (auto& changed_cell : changed) {
		if (_game_field[changed_cell.y][changed_cell.x]) {
			_game_field[changed_cell.y][changed_cell.x] = false;

			_alive_cells.erase(changed_cell);
		}
		else {
			_game_field[changed_cell.y][changed_cell.x] = true;
			_alive_cells.insert(changed_cell);
		}
	}
}



void set_range_change(const coord_t& cell_coord, tmp_coord_t& c_min, tmp_coord_t& c_max, const size_t& max_value) {
	if (cell_coord == 0) {
		c_min = 0;
		c_max = 1;
	}
	else if (cell_coord > 0 && cell_coord != max_value - 1) {
		c_min = -1;
		c_max = 1;
	}
	else {//when cell_coord == max_value - 1
		c_min = -1;
		c_max = 0;
	}
}

void set_change_list(std::set<cell_coord>& list, std::vector<cell_coord>& _to_change, std::vector<std::vector<bool>>& _game_field, const bool _add_died_cells_check, std::set<cell_coord>& _died_cells_check)
{
	for (auto& alive_cell : list) {
		int count_alive_cells = 0;
		coord_t cell_y = alive_cell.y;
		coord_t cell_x = alive_cell.x;

		tmp_coord_t y_min, y_max;
		tmp_coord_t x_min, x_max;


		set_range_change(cell_y, y_min, y_max, MAX_GAME_FIELD_Y);
		set_range_change(cell_x, x_min, x_max, MAX_GAME_FIELD_X);



		for (tmp_coord_t y_tmp = y_min; y_tmp <= y_max; ++y_tmp) {
			for (tmp_coord_t x_tmp = x_min; x_tmp <= x_max; ++x_tmp) {
				coord_t y = cell_y + y_tmp;
				coord_t x = cell_x + x_tmp;

				bool _is_alive = _game_field[y][x];
				if (_is_alive) {
					count_alive_cells++;
				}
				else {
					if (_add_died_cells_check) {
						_died_cells_check.insert({ y,x });
					}
				}

			}
		}

		if (_add_died_cells_check) {
			if (count_alive_cells != 3 && count_alive_cells != 4) {
				_to_change.push_back(alive_cell);
			}
		}
		else {
			if (count_alive_cells == 3) {
				_to_change.push_back(alive_cell);
			}
		}
	}
}


void game::setCell(coord_t x, coord_t y)
{
	if (x < _game_field.size() && y < _game_field.begin()->size())
	{
		bool cell = !_game_field[x][y];
		_game_field[x][y] = cell;
		_alive_cells.insert({ x, y });
	}
}

const std::set<cell_coord>& game::getAliveCells() const
{
	return _alive_cells;
}