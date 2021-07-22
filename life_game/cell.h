#pragma once
#include "const_values.h"	

	class cell
	{
	public:
		cell(const short y, const short x, const bool alive = false);

		const bool _u_alive() const;

		void _kill();

		void _resurrect();

		const coord_t _show_y_coord() const;

		const coord_t _show_x_coord() const;

	private:
		const coord_t _y_coord;
		const coord_t _x_coord;
		bool _alive;
	};

	struct cell_coord {
		coord_t y;
		coord_t x;
	};

#define d_op(OP)\
	bool operator OP (const cell_coord& lhs, const cell_coord& rhs);

d_op(< );
d_op(> );
d_op(<= );
d_op(>= );
d_op(!= );
d_op(== );



#undef d_op