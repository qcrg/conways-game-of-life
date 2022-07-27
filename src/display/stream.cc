#include "stream.hpp"

namespace gol::display
{

    stream::stream(uint x, uint y, std::ostream &op)
        : x_size{x}
        , y_size{y}
        , output{op}
    {}

    
    void stream::present(const field_t &field,
            uint x_offset, uint y_offset)
    {
        for (uint y = y_offset; y < y_size; y++)
        {
            for (uint x = x_offset; x < x_size; x++)
            {
                output << (field[point_t{x, y}] ? '#' : ' ');
            }
            if (y != y_size - 1)
                output << '\n';
        }
        output << std::flush;
    }
    

} //namespace gol
