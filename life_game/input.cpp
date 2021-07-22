#include "input.h"
#include "const_values.h"

#include <sstream>
#include <iostream>

input::input(std::string name_file)
{
	std::ifstream in_file(name_file);
	if (!in_file) {
		std::cerr << "input not found(\"game_info\")\n";
	}
	std::string l;

	coord_t y = 0;
	coord_t x = 0;
	while (std::getline(in_file, l)) {
		std::istringstream line(l);
		line >> std::ws;
		char code[4];
		line.get(code, 4);

		if (code[0] == '\\' && code[1] == '-' && code[2] == '>') {
			for (y = 0; y < MAX_GAME_FIELD_Y; ++y) {
				char c;
				line.get(c);
				if (c == '1') {
					coords_alive_cels.insert({ y, x });
				}
				else if(c == '0') {
					continue;
				}
				else if(c == '/'){
					break;
				}
			}
			++x;
		}
	}


}

const std::set<std::pair<coord_t, coord_t>>& input::get_alive_cels() const
{
	return coords_alive_cels;
}
