#pragma once

#include "field.hxx"

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
        { t.get_field() } -> std::same_as<FieldConc>;
    };

    template<FieldConc F>
    class EngineBasic
    {
    public:
        using FieldType = F;
    private:
        FieldType field;
        unsigned int tps;
        std::atomic<bool> is_play, is_one_step, quit_from_loop;
        std::thread thread;
        std::mutex mutex;

        void tick();
    public:
        EngineBasic();

        void run();
        void quit();

        bool is_played();
        void play();
        void pause();
        void one_step();
        void set_speed(unsigned int ticks_per_second);
        unsigned int get_speed() const;
        void change_cell(const Point &point);
        
        const FieldType &get_field() const;
    };

    template<FieldConc F>
    EngineBasic<F>::EngineBasic()
        : tps{4}
        , is_play{false}
        , is_one_step{false}
        , quit_from_loop{false}
    {
        (void)mutex.try_lock();
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
                int64_t mil =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock::now() - last_tick_time).count();
                std::scoped_lock{mutex};
                if (is_play && (1000 / tps) < mil)
                {
                    last_tick_time = clock::now();
                    this->tick();
                }
                else if (is_one_step)
                {
                    this->tick();
                    is_one_step = false;
                    (void)mutex.try_lock();
                }
                else
                {
                    auto rest_time =
                        std::chrono::milliseconds((1000 / tps) - mil);
                    std::this_thread::sleep_for(rest_time);
                }
            }
        };
        thread = std::thread(func);
        thread.join();
    }

    template<FieldConc F>
    void EngineBasic<F>::quit()
    {
        mutex.unlock();
        quit_from_loop = true;
    }

    template<FieldConc F>
    void EngineBasic<F>::play()
    {
        mutex.unlock();
        is_play = true;
    }

    template<FieldConc F>
    void EngineBasic<F>::pause()
    {
        (void)mutex.try_lock();
        is_play = false;
    }

    template<FieldConc F>
    void EngineBasic<F>::one_step()
    {
        mutex.unlock();
        is_one_step = true;
    }

    template<FieldConc F>
    void EngineBasic<F>::set_speed(unsigned int tps)
    {
        this->tps = tps;
    }

    template<FieldConc F>
    unsigned int EngineBasic<F>::get_speed() const
    {
        return tps;
    }

    template<FieldConc F>
    void EngineBasic<F>::change_cell(const Point &point)
    {
        field.change(point);
    }

    template<FieldConc F>
    void EngineBasic<F>::tick()
    {
        const auto &alive = field.get_alive();
        std::unordered_set<Point> for_change;
        auto is_for_change = [&](const Point &p)
        {
            int alive_count = 0;
            for (int i = -1; i < 2; i++)
                for (int j = -1; j < 2; j++)
                    if (field.is_alive({static_cast<dim_t>(p.x + i),
                                static_cast<dim_t>(p.y + j)}))
                        alive_count++;
            return field.is_alive(p) ?
                !(alive_count == 3 || alive_count == 4) :
                alive_count == 3;
        };
        for (const auto &p : alive)
        {
            for (int i = -1; i < 2; i++)
                for (int j = -1; j < 2; j++)
                    if (is_for_change({static_cast<dim_t>(p.x + i),
                                static_cast<dim_t>(p.y + j)}))
                        for_change.insert({static_cast<dim_t>(p.x + i),
                                static_cast<dim_t>(p.y + j)});
        }
        for (const auto &p : for_change)
            field.change(p);
    }

    template<FieldConc F>
    const F &EngineBasic<F>::get_field() const
    {
        return field;
    }

    using Engine = EngineBasic<Field>;
} //namespace pnd::gol
