#include <iostream>
#include <thread>
#include <chrono>

#include "input.h"
#include "game.h"
#include "output.h"


int main() {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);

	input in("game_info");
	game game(in.get_alive_cels());
	output out(game._show_game_field());
	while (true)
	{
		out._show_game();
		game._one_beat();
	}
	
	std::cerr << "the end\n";
	return 0;
}