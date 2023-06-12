#include "field.hxx"

#include <stdexcept>

namespace pnd::gol
{

    void Field::add(const Point &p)
    {
        auto it = data.insert(p);
        if (!it.second)
            throw std::invalid_argument(
                    "Cant be added: the Point has already been added");
    }

    void Field::remove(const Point &p)
    {
        auto it = data.find(p);
        if (it != data.end())
            data.erase(it);
        else
            throw std::invalid_argument("Cant be removed: there is not Point");
    }

    void Field::change(const Point &p)
    {
        auto it = data.insert(p);
        if (!it.second)
            data.erase(it.first);
    }

    const Alives &Field::get_alives() const
    {
        return data;
    }

    bool Field::is_alive(const Point &p) const
    {
        return data.contains(p);
    }

} //namespace pnd::gol
