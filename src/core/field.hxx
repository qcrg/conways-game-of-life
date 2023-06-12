#pragma once

#include "point.hxx"

#include <unordered_set>
#include <concepts>

namespace pnd::gol
{

    template<typename T>
    concept FieldConc = requires(T t)
    {
        t.add(Point{});
        t.remove(Point{});
        t.change(Point{});
        { t.is_alive(Point()) } -> std::same_as<bool>;
        { t.get_alives() } -> std::same_as<const Alives &>;
    };
    
    class Field
    {
        Alives data;

    public:
        void add(const Point &p);
        void remove(const Point &p);
        void change(const Point &p);
        bool is_alive(const Point &p) const;
        const Alives &get_alives() const;

    };

} //namespace pnd::gol
