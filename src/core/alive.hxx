#pragma once

#include "point.hxx"

#include <unordered_set>
#include <iterator>

namespace pnd::gol
{
    template<typename T>
    concept AliveConc = requires(T t)
    {
        t.add(Point());
        t.remove(Point());
        { t.begin() } -> std::same_as<std::forward_iterator>;
        { t.end() } -> std::same_as<std::forward_iterator>;
        { t.is_alive(Point()) } -> std::same_as<bool>;
    };

    class Alive
    {
        std::unordered_set<Point> data;
    public:
        auto begin();
        auto begin() const;
        auto end();
        auto end() const;
        void add(const Point &p);
        void remove(const Point &p);
        bool is_alive(const Point &p) const;
        bool operator[](const Point &p) const;
    };

    auto Alive::begin()
    {
        return data.begin();
    }

    auto Alive::begin() const
    {
        return data.begin();
    }

    auto Alive::end()
    {
        return data.end();
    }

    auto Alive::end() const
    {
        return data.end();
    }

    void Alive::add(const Point &p)
    {
        data.insert(p);
    }

    void Alive::remove(const Point &p)
    {
        data.erase(p);
    }

    bool Alive::is_alive(const Point &p) const
    {
        return data.contains(p);
    }

    bool Alive::operator[](const Point &p) const
    {
        return is_alive(p);
    }

} //namespace pnd::gol
