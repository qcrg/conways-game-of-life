#pragma once
#include <vector>
#include <iostream>

	class screen
	{
	public:
		screen();


		friend std::ostream& operator<< (std::ostream& os, const screen& scr);

	private:
		std::vector<std::vector<bool>> data;
		static const unsigned int cols = 120;
		static const unsigned int rows = 30;
	};

	std::ostream& operator<< (std::ostream& os, const screen& scr);

