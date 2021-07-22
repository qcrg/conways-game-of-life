#include "screen.h"

std::ostream& operator<<(std::ostream& os, const screen& scr)
{
	for (unsigned int i = 0; i < scr.rows; ++i) {
		for (unsigned int ii = 0; ii < scr.cols; ++ii) {
			std::cout << (scr.data[ii][i] ? "0" : " ");
		}
		std::cout << '\n';
	}

	return os;
}

screen::screen()
{
	data.resize(cols);
	for (int i = 0; i < cols; ++i) {
		data[i].resize(rows, false);
	}
}
