#pragma once

#include <set>
#include <vector>

#include "game.h"

class output
{
public:
	output(const std::vector<std::vector<bool>>& _game_field);

	void _show_game();

private:
	const std::vector<std::vector<bool>>& _game_field;
};

