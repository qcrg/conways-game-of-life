#pragma once
#include "const_values.h"

#include <fstream>
#include <set>
#include <string>



class input
{
public:
	input(std::string name_file);

	const std::set<std::pair<coord_t, coord_t>>& get_alive_cels() const;

private:
	std::set<std::pair<coord_t, coord_t>> coords_alive_cels;
};

