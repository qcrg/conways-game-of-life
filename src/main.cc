#include <thread>
#include <iostream>

#include "core/engine.hpp"
#include "display/stream.hpp"

constexpr uint size_x = 80;
constexpr uint size_y = 24;

void add_cells(gol::engine &engine)
{
    engine.change_cell({0, 0});
    engine.change_cell({0, 1});
    engine.change_cell({1, 0});

    engine.change_cell({6, 5});
    engine.change_cell({6, 6});
    engine.change_cell({6, 7});
}

int main(int argc, char **argv)
{
    argc = argc; argv = argv;
    gol::engine engine(size_x, size_y);
    gol::display::stream viewer(size_x, size_y, std::cout);
    add_cells(engine);
    while (true)
    {
        viewer.present(engine.get_field());
        engine.tick();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}
