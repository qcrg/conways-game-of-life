#pragma once

#include "point.hxx"
#include "alive.hxx"

#include <unordered_set>
#include <concepts>
#include <vector>
#include <cmath>
#include <limits>

namespace pnd::gol
{

    template<typename T>
    concept FieldConc = requires(T t)
    {
        t.add(Point{});
        t.remove(Point{});
        t.change(Point{});
        { t.is_alive(Point()) } -> std::same_as<bool>;
        { t.get_alive() } -> std::same_as<AliveConc>;
    };
    
    template<AliveConc T>
    class FieldBasic
    {
        using AliveType = T;
        AliveType alive;
        std::vector<bool> data;
        std::vector<bool>::reference get(const Point &p);
        bool get(const Point &p) const;
    public:
        FieldBasic();
        void add(const Point &p);
        void remove(const Point &p);
        void change(const Point &p);
        bool is_alive(const Point &) const;
        const AliveType &get_alive() const;
    };

    template<AliveConc T>
    FieldBasic<T>::FieldBasic()
        : data(std::pow(dim_t_size, 2), false)
    {
    }

    template<AliveConc T>
    void FieldBasic<T>::add(const Point &p)
    {
        get(p) = true;
        alive.add(p);
    }

    template<AliveConc T>
    void FieldBasic<T>::remove(const Point &p)
    {
        get(p) = false;
        alive.remove(p);
    }

    template<AliveConc T>
    void FieldBasic<T>::change(const Point &p)
    {
        get(p) = !get(p);
        get(p) ? alive.add(p) : alive.remove(p);
    }

    template<AliveConc T>
    bool FieldBasic<T>::is_alive(const Point &p) const
    {
        return get(p);
    }

    template<AliveConc T>
    const T &FieldBasic<T>::get_alive() const
    {
        return alive;
    }

    template<AliveConc T>
    std::vector<bool>::reference FieldBasic<T>::get(const Point &p)
    {
        return data[dim_t_size * p.y + p.x];
    }

    template<AliveConc T>
    bool FieldBasic<T>::get(const Point &p) const
    {
        return data[dim_t_size * p.y + p.x];
    }

    using Field = FieldBasic<Alive>;

} //namespace pnd::gol
