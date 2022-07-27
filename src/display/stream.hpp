#pragma once
#include "core/engine.hpp"
#include <ostream>

namespace gol::display
{
    struct stream
    {
        stream(uint x_size, uint y_size, std::ostream &output);
        void present(const field_t &field,
                uint x_offset = 0, uint y_offset = 0);
        void set_display_size(uint x_size, uint y_size);
    private:
        uint x_size;
        uint y_size;
        std::ostream &output;
    };
} //namespace gol::display
