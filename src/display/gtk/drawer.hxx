#pragma once

#include "core/engine.hxx"
#include "core/point.hxx"

#include <cairomm/context.h>
#include <gtkmm/drawingarea.h>
#include <gtkmm/scrollable.h>
#include <gtkmm/actionable.h>
#include <gtkmm/eventcontrollermotion.h>
#include <gtkmm/eventcontrollerscroll.h>
#include <gtkmm/adjustment.h>
#include <gtkmm/gestureclick.h>
#include <glibmm/main.h>
#include <glibmm/init.h>

//FIXME
#include <iostream>
#include <format>

namespace pnd::gol
{
    struct DrawerData
    {
        constexpr static int size_min = 4;
        constexpr static int size_max = 20;
        constexpr static int size_diff = 1;
        constexpr static float offset_mult = 1.0;
        bool middle_mouse_pressed = false;
        float offset_x = 0;
        float offset_y = 0;
        int size = 10;
        float last_mouse_x = 0;
        float last_mouse_y = 0;
    };

    template<EngineConc E>
    class DrawerBasic : public std::enable_shared_from_this<DrawerBasic<E>>
    {
        using ThisType = DrawerBasic<E>;
    public:
        using EngineType = E;
    private:
        DrawerData ctx;
        EngineType &engine;
        Glib::RefPtr<Gtk::DrawingArea> draw_area;
        Glib::RefPtr<Gtk::EventControllerMotion> motion_controller;
        Glib::RefPtr<Gtk::EventControllerScroll> scroll_controller;
        Glib::RefPtr<Gtk::GestureClick> left_button_controller;
        Glib::RefPtr<Gtk::GestureClick> middle_button_controller;

        void operator()(const Cairo::RefPtr<Cairo::Context> &cr, int w, int h);
        void process_motion(double x, double y);
        bool process_scroll(double dx, double dy);
        void process_left_button(int, double x, double y, bool pressed);
        void process_middle_button(int, double x, double y, bool pressed);
        std::pair<float, float> get_mouse_idx_float() const;
        Point get_mouse_idx() const;

        DrawerBasic(EngineType &engine,
                Glib::RefPtr<Gtk::DrawingArea> draw_area);
    public:
        static Glib::RefPtr<ThisType> create(EngineType &engine,
                Glib::RefPtr<Gtk::DrawingArea> draw_area);
        ~DrawerBasic();

        void left();
        void right();
        void up();
        void down();
    };

    template<EngineConc E>
    DrawerBasic<E>::DrawerBasic(EngineType &engine,
            Glib::RefPtr<Gtk::DrawingArea> draw_area_)
        : engine{engine}
        , draw_area{std::move(draw_area_)}
    {
        Glib::init();
        motion_controller = Gtk::EventControllerMotion::create();
        scroll_controller = Gtk::EventControllerScroll::create();
        left_button_controller = Gtk::GestureClick::create();
        middle_button_controller = Gtk::GestureClick::create();

        using namespace std::placeholders;
        auto draw_func = std::bind(&DrawerBasic<E>::operator(), this,
                _1, _2, _3);
        auto update_func = [this] {
            this->draw_area->queue_draw();
            return true;
        };

        draw_area->set_draw_func(draw_func);

        draw_area->add_controller(motion_controller);
        motion_controller->signal_motion().connect(
                std::bind(&ThisType::process_motion, this, _1, _2));

        draw_area->add_controller(scroll_controller);
        scroll_controller->signal_scroll().connect(
                sigc::mem_fun(*this, &ThisType::process_scroll), true);
        scroll_controller->set_flags(
                Gtk::EventControllerScroll::Flags::VERTICAL);

        draw_area->add_controller(left_button_controller);
        left_button_controller->set_button(GDK_BUTTON_PRIMARY);
        left_button_controller->signal_pressed().connect(
                std::bind(&ThisType::process_left_button, this,
                    _1, _2, _3, true));
        left_button_controller->signal_released().connect(
                std::bind(&ThisType::process_left_button, this,
                    _1, _2, _3, false));

        draw_area->add_controller(middle_button_controller);
        middle_button_controller->set_button(GDK_BUTTON_MIDDLE);
        middle_button_controller->signal_pressed().connect(
                std::bind(&ThisType::process_middle_button, this,
                    _1, _2, _3, true));
        middle_button_controller->signal_released().connect(
                std::bind(&ThisType::process_middle_button, this,
                    _1, _2, _3, false));

        draw_area->set_can_focus();
        draw_area->set_focusable();
        draw_area->set_focus_on_click();

        Glib::signal_timeout().connect(update_func, 17);
    }

    template<EngineConc E>
    DrawerBasic<E>::~DrawerBasic()
    {
    }

    template<EngineConc E>
    void DrawerBasic<E>::operator()(const Cairo::RefPtr<Cairo::Context> &cr,
            int w, int h)
    {
        cr->set_source_rgb(0, 0, 0);
        cr->paint();
        for (int i = ctx.offset_x - 1;
                i < w / ctx.size + ctx.offset_x;
                i++)
            for (int j = ctx.offset_y - 1;
                    j < h / ctx.size + ctx.offset_y;
                    j++)
            {
                if (engine.get_field().is_alive({static_cast<dim_t>(i),
                            static_cast<dim_t>(j)}))
                {
                    float bord = std::round(ctx.size / 10),
                        x_idx = i - ctx.offset_x,
                        y_idx = j - ctx.offset_y,
                        x = x_idx * ctx.size + bord,
                        y = y_idx * ctx.size + bord;
                    int size = ctx.size - static_cast<int>(bord);
                    cr->rectangle(static_cast<int>(x),
                            static_cast<int>(y),
                            size, size);
                }
            }
        cr->set_source_rgb(255, 0, 0);
        cr->fill();
    }

    template<EngineConc E>
    void DrawerBasic<E>::process_motion(double x, double y)
    {
        if (ctx.middle_mouse_pressed)
        {
            ctx.offset_x -= (x - ctx.last_mouse_x) * ctx.offset_mult / ctx.size;
            ctx.offset_y -= (y - ctx.last_mouse_y) * ctx.offset_mult / ctx.size;
        }
        ctx.last_mouse_x = x;
        ctx.last_mouse_y = y;
    }

    template<EngineConc E>
    bool DrawerBasic<E>::process_scroll(double, double dy)
    {
        if (!ctx.middle_mouse_pressed)
        {
            std::pair<float, float> old_idx = get_mouse_idx_float();
            int old_size = ctx.size;
            ctx.size = std::clamp(ctx.size - static_cast<int>(dy) * ctx.size_diff,
                    ctx.size_min,
                    ctx.size_max);
            std::pair<float, float> new_idx = get_mouse_idx_float();
            if (old_size != ctx.size)
            {
                ctx.offset_x += old_idx.first - new_idx.first;
                ctx.offset_y += old_idx.second - new_idx.second;
            }
        }
        return true;
    }

    template<EngineConc E>
    std::pair<float, float>
    DrawerBasic<E>::get_mouse_idx_float() const
    {
        return {
            ctx.last_mouse_x / ctx.size + ctx.offset_x,
            ctx.last_mouse_y / ctx.size + ctx.offset_y
        };
    }

    template<EngineConc E>
    Point DrawerBasic<E>::get_mouse_idx() const
    {
        auto idx = get_mouse_idx_float();
        float x = idx.first,
              y = idx.second;
        return {
            static_cast<dim_t>(x < 0 ? x - 1 : x),
            static_cast<dim_t>(y < 0 ? y - 1 : y)
        };
    }

    template<EngineConc E>
    void DrawerBasic<E>::process_left_button(int, double, double,
            bool pressed)
    {
        if (pressed)
        {
            engine.change_cell(get_mouse_idx());
            left_button_controller->set_state(Gtk::EventSequenceState::CLAIMED);
        }
    }

    template<EngineConc E>
    void DrawerBasic<E>::process_middle_button(int, double, double,
            bool pressed)
    {
        ctx.middle_mouse_pressed = pressed;
        middle_button_controller->set_state(Gtk::EventSequenceState::CLAIMED);
    }

    template<EngineConc E>
    Glib::RefPtr<DrawerBasic<E>> DrawerBasic<E>::create(EngineType &engine,
            Glib::RefPtr<Gtk::DrawingArea> draw_area)
    {
        return Glib::RefPtr<ThisType>(new ThisType(engine, std::move(draw_area)));
    }

    template<EngineConc E>
    void DrawerBasic<E>::left()
    {
        ctx.offset_x--;
    }

    template<EngineConc E>
    void DrawerBasic<E>::right()
    {
        ctx.offset_x++;
    }

    template<EngineConc E>
    void DrawerBasic<E>::up()
    {
        ctx.offset_y--;
    }

    template<EngineConc E>
    void DrawerBasic<E>::down()
    {
        ctx.offset_y++;
    }

    using Drawer = DrawerBasic<Engine>;

} //namespace pnd::gol
