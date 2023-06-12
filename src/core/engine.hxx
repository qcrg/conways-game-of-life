#pragma once

#include "field.hxx"

#include <sigc++/sigc++.h>
#include <concepts>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>

namespace pnd::gol
{
    template<typename T>
    concept EngineConc = requires(T t)
    {
        t.run();
        t.quit();
        { t.is_played() } -> std::same_as<bool>;
        t.play();
        t.pause();
        t.one_step();
        t.set_speed(int());
        t.change_cell(Point());
        std::is_same_v<sigc::signal<void(Alives)>,
            decltype(t.tick_ended)>;
    };

    template<FieldConc F>
    class EngineBasic
    {
        using FieldType = F;

        FieldType field;
        int tps;
        std::atomic<bool> is_play, is_one_step, quit_from_loop;
        std::mutex mutex;
        std::thread thread;

        void tick();

    public:
        EngineBasic();

        void run();
        void quit();

        bool is_played();
        void play();
        void pause();
        void one_step();
        void set_speed(int ticks_per_second);
        void change_cell(const Point &point);

        sigc::signal<void(const Alives &)> tick_ended;

    };

    template<FieldConc F>
    EngineBasic<F>::EngineBasic()
        : tps{4}
        , is_play{true}
        , is_one_step{false}
        , quit_from_loop{false}
    {
    }

    template<FieldConc F>
    bool EngineBasic<F>::is_played()
    {
        return is_play;
    }

    template<FieldConc F>
    void EngineBasic<F>::run()
    {
        auto func = [&]()
        {
            using clock = std::chrono::steady_clock;
            auto last_tick_time = clock::now();
            while (!quit_from_loop)
            {
                auto mil =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock::now() - last_tick_time).count();
                if (is_play && (1000 / tps) < mil)
                {
                    last_tick_time = clock::now();
                    this->tick();
                }
                else if (is_one_step)
                {
                    this->tick();
                    is_one_step = false;
                }
            }
        };
        thread = std::thread(func);
        thread.join();
    }

    template<FieldConc F>
    void EngineBasic<F>::quit()
    {
        quit_from_loop = true;
    }

    template<FieldConc F>
    void EngineBasic<F>::play()
    {
        is_play = true;
    }

    template<FieldConc F>
    void EngineBasic<F>::pause()
    {
        is_play = false;
    }

    template<FieldConc F>
    void EngineBasic<F>::one_step()
    {
        is_one_step = true;
    }

    template<FieldConc F>
    void EngineBasic<F>::set_speed(int tps)
    {
        this->speed = tps;
    }

    template<FieldConc F>
    void EngineBasic<F>::change_cell(const Point &point)
    {
        std::scoped_lock<std::mutex> lock(mutex);
        field.change(point);
    }

    template<FieldConc F>
    void EngineBasic<F>::tick()
    {
        const auto &alive = field.get_alives();
        std::unordered_set<Point> for_change;
        auto is_for_change = [&](const Point &p)
        {
            int alive_count = 0;
            for (int i = -1; i < 2; i++)
                for (int j = -1; j < 2; j++)
                    if (field.is_alive({p.x + i, p.y + j}))
                        alive_count++;
            bool b = alive_count == 3 || alive_count == 4;
            return field.is_alive(p) ? !b : b;
            //bool a = field.is_alive(p),
                 //b = alive_count == 3 || alive_count == 4;
            //return (a || b) && (!a || b);
        };
        std::scoped_lock lock(mutex);
        for (const auto &p : alive)
        {
            for (int i = -1; i < 2; i++)
                for (int j = -1; j < 2; j++)
                    if (is_for_change({p.x + i, p.y + j}))
                        for_change.insert({p.x + i, p.y + j});
        }
        for (const auto &p : for_change)
            field.change(p);
        tick_ended.emit(field.get_alives());
    }

    using Engine = EngineBasic<Field>;
} //namespace pnd::gol
