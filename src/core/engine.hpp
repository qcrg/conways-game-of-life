#pragma once

#include <unordered_set>
#include <array>
#include <cstddef>

namespace gol
{

    using uint = unsigned int;

    struct point_t
    {
        uint x;
        uint y;
        bool operator==(const point_t &r) const;
    };

    struct point_hash
    {
        size_t operator()(const point_t &p) const;
    };

    using cells_pack = std::unordered_set<point_t, point_hash>;

    struct field_t
    {
        bool *begin;
        bool *end;
        uint x_size;
        uint y_size;

        field_t(uint x_size, uint y_size);

        ~field_t();

        bool &operator[](const point_t &p);
        const bool &operator[](const point_t &p) const;
    };

    struct engine
    {
        engine(uint x_size, uint y_size);
        void tick();
        const bool &change_cell(const point_t &p);
        const bool &set_cell(const point_t &p, bool alive);
        const bool &get_cell(const point_t &p) const;
        const cells_pack &get_alive_cells() const;
        const field_t &get_field() const;
    private:
        field_t field;
        cells_pack alive_cells;

        bool allowable(const point_t &p) const;
        std::array<point_t, 9> get_square_3x3(point_t p) const;
        bool need_to_change_cell(const point_t &cell) const;
        cells_pack get_cells_for_change(const point_t &alive_cell) const;
        cells_pack handle_alive_cells(const cells_pack &alive_targets) const;

    };


} //namespace gol
