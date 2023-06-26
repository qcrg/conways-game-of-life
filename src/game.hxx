#pragma once

#include "core/engine.hxx"
#include "display/display.hxx"
#include "window.hxx"

namespace pnd::gol
{
    template<typename T>
    concept GameConc = requires(T t)
    {
        (void)t;
    };

    template
    <
        AliveConc A = Alive,
        FieldConc F = FieldBasic<A>,
        EngineConc E = EngineBasic<F>,
        DisplayConc D = WindowBasic<E>
    >
    class GameBasic
    {
        E engine;
        D display;

    public:
        using AliveType = A;
        using FieldType = F;
        using EngineType = E;
        using DisplayType = D;

        GameBasic();
        ~GameBasic();

    };

    template<AliveConc A, FieldConc F, EngineConc E, DisplayConc D>
    GameBasic<A, F, E, D>::GameBasic()
        : display{engine}
    {
        //FIXME delete change cells
        engine.change_cell({0, 0});
        engine.change_cell({0, 1});
        engine.change_cell({1, 0});
        engine.change_cell({6, 5});
        engine.change_cell({6, 6});
        engine.change_cell({6, 7});

        engine.run();
    }

    template<AliveConc A, FieldConc F, EngineConc E, DisplayConc D>
    GameBasic<A, F, E, D>::~GameBasic()
    {
        engine.quit();
    }

    using Game = GameBasic<Alive, Field, Engine, Window>;

} //namespace pnd::gol
