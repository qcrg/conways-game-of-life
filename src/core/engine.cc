#include "engine.hpp"

#include <array>
#include <unordered_set>
#include <cstddef>
#include <functional>

namespace gol
{

    bool point_t::operator==(const point_t &r) const
    {
        return std::make_tuple(x, y) == std::make_tuple(r.x, r.y);
    }

    size_t point_hash::operator()(const point_t &p) const
    {
        return static_cast<size_t>(p.x) << (sizeof(size_t) * 4)
            | static_cast<size_t>(p.y);
    }

    field_t::field_t(uint x_size, uint y_size)
        : begin{new bool[x_size * y_size]}
        , end{begin + x_size + y_size}
        , x_size{x_size}
        , y_size{y_size}
    {}

    field_t::~field_t()
    {
        delete[] begin;
    }

    bool &field_t::operator[](const point_t &p)
    {
        return begin[p.y * x_size + p.x];
    }

    const bool &field_t::operator[](const point_t &p) const
    {
        return begin[p.y * x_size + p.x];
    }

    engine::engine(uint x_size, uint y_size)
        : field{x_size, y_size}
    {
        std::fill(field.begin, field.end, 0);
    }

    void engine::tick()
    {
        for (point_t cell : handle_alive_cells(alive_cells))
            change_cell(std::move(cell));
    }

    const bool &engine::change_cell(const point_t &p)
    {
        bool &cell = field[p];
        if (cell)
            alive_cells.erase(p);
        else
            alive_cells.insert(std::move(p));
        cell = !cell;
        return cell;
    }

    const bool &engine::set_cell(const point_t &p, bool alive)
    {
        bool &cell = field[p];
        if (cell != alive)
            change_cell(std::move(p));
        return cell;
    }

    const bool &engine::get_cell(const point_t &p) const
    {
        return field[p];
    }

    const cells_pack &engine::get_alive_cells() const
    {
        return alive_cells;
    }

    const field_t &engine::get_field() const 
    {
        return field;
    }

    bool engine::allowable(const point_t &p) const
    {
        return p.x > 0 && p.x < field.x_size
            && p.y > 0 && p.y < field.y_size;
    }

    std::array<point_t, 9> engine::get_square_3x3(point_t p) const
    {
        std::array<point_t, 9> res;
        p.x -= 1;
        p.y -= 1;

        for (int y : {0, 1, 2})
            for (int x : {0, 1, 2})
                res[y * 3 + x] = {p.x + x, p.y + y};
        return res;
    }

    bool engine::need_to_change_cell(const point_t &cell) const
    {
        int alive_cells = 0;
        for (auto &cell : get_square_3x3(cell))
            if (field[cell])
                alive_cells++;
        return field[cell] ^ (alive_cells == 3 || alive_cells == 4);
    }

    cells_pack engine::get_cells_for_change(const point_t &alive_cell) const
    {
        cells_pack res;
        for (const auto &check_cell : get_square_3x3(alive_cell))
            if (allowable(check_cell) && need_to_change_cell(check_cell))
                res.insert(check_cell);
        return res;
    }

    cells_pack engine::handle_alive_cells(const cells_pack &alive_targets) const
    {
        cells_pack res;
        for (const point_t &target : alive_targets)
            res.merge(get_cells_for_change(target));
        return res;
    }

} //namespace gol
