#include "output.h"
#include "const_values.h"

#include <iostream>
#include <fstream>

output::output(const std::vector<std::vector<bool>>& game_field) : _game_field(game_field)
{
}

void output::_show_game()
{
	system("cls");
	for (coord_t x = 0; x < MAX_GAME_FIELD_X; ++x) {
		for (coord_t y = 0; y < MAX_GAME_FIELD_Y; ++y) {
			
			std::cout << (_game_field[y][x] ? '0' : ' ');
		}
		std::cout << '\n';
	}
}
