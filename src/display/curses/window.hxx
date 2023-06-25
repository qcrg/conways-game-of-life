#pragma once

#include "core/engine.hxx"

#include <curses.h>
#include <thread>
#include <mutex>
#include <cmath>

namespace pnd::gol
{
    template<EngineConc E = Engine>
    class WindowBasic
    {
        using ThisType = WindowBasic<E>;
        static const char alive_char_present = '#';
        int offset_x, offset_y;
        int cursor_x, cursor_y;
        bool insert_mode;
        WINDOW *wnd;
        std::jthread thread;
        std::mutex mutex;
        E &engine;

        void process_input();
        void process_thread(std::stop_token token);
        void render();
        
    public:
        using EngineType = E;

        WindowBasic(EngineType &engine);
        ~WindowBasic();
    };

    template<EngineConc E>
    WindowBasic<E>::WindowBasic(E &engine)
        : offset_x{0}
        , offset_y{0}
        , cursor_x{0}
        , cursor_y{0}
        , insert_mode{false}
        , wnd{initscr()}
        , engine{engine}
    {
        setlocale(LC_ALL, "");
        cbreak();
        noecho();
        timeout(10);
        curs_set(0);
        thread = std::jthread(std::bind(&ThisType::process_thread,
                    this,
                    std::placeholders::_1));
    }

    template<EngineConc E>
    void WindowBasic<E>::process_thread(std::stop_token token)
    {
        while (!token.stop_requested())
        {
            process_input();
            refresh();
            clear();
            render();
            if (insert_mode)
                wmove(wnd, cursor_y, cursor_x);
        }
    }

    template<EngineConc E>
    void WindowBasic<E>::render()
    {
        std::scoped_lock<std::mutex> lock(mutex);
        int term_x, term_y;
        getmaxyx(wnd, term_y, term_x);
        for (int i = offset_x; i < offset_x + term_x; i++)
            for (int j = offset_y; j < offset_y + term_y; j++)
                if (engine.get_field().is_alive({static_cast<dim_t>(i),
                            static_cast<dim_t>(j)}))
                    mvwaddch(wnd,
                            j - offset_y, i - offset_x,
                            alive_char_present);
    }

    template<EngineConc E>
    void WindowBasic<E>::process_input()
    {
        switch (wgetch(wnd))
        {
            case KEY_LEFT:
            case 'h':
                if (insert_mode)
                    cursor_x--;
                else
                    offset_x--;
            break;
            
            case KEY_DOWN:
            case 'j':
                if (insert_mode)
                    cursor_y++;
                else
                    offset_y++;
            break;

            case KEY_UP:
            case 'k':
                if (insert_mode)
                    cursor_y--;
                else
                    offset_y--;
            break;

            case KEY_RIGHT:
            case 'l':
                if (insert_mode)
                    cursor_x++;
                else
                    offset_x++;
            break;

            case 'p':
                if (engine.is_played())
                    engine.pause();
                else
                    engine.play();
            break;

            case 'q':
                engine.quit();
            break;

            case 'o':
                engine.one_step();
            break;

            case 'i':
                if (!insert_mode)
                {
                    insert_mode = true;
                    curs_set(2);
                }
            break;

            case ' ':
                if (insert_mode)
                {
                    engine.change_cell(
                            {
                                static_cast<dim_t>(cursor_x + offset_x),
                                static_cast<dim_t>(cursor_y + offset_y)
                            });
                }
            break;

            case 27:
                if (insert_mode)
                {
                    insert_mode = false;
                    curs_set(0);
                }
            break;

            case '-':
                engine.set_speed(std::max(engine.get_speed() - 1, 1u));
            break;

            case '+':
            case '=':
                engine.set_speed(std::min(engine.get_speed() + 1, 10000u));
            break;

        };
    }

    template<EngineConc E>
    WindowBasic<E>::~WindowBasic()
    {
        thread.request_stop();
        thread.join();
        endwin();
    }

    using Window = WindowBasic<Engine>;
} //namespace pnd::gol
