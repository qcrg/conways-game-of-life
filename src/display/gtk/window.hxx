#pragma once

#include "drawer.hxx"
#include "core/engine.hxx"

#include <thread>
#include <gtkmm/application.h>
#include <gtkmm/builder.h>
#include <gtkmm/button.h>
#include <gtkmm/window.h>
#include <gtkmm/spinbutton.h>
#include <gtkmm/box.h>
#include <gtkmm/drawingarea.h>
#include <stdexcept>

constexpr const char interface_name[] = "gen/interface.ui";

namespace pnd::gol
{
    template<EngineConc E, typename D = DrawerBasic<E>>
    class WindowBasic
    {
        using ThisType = WindowBasic<E, D>;
        using DrawerType = D;
    public:
        using EngineType = E;
    private:
        EngineType &engine;
        std::thread thread;
        Glib::RefPtr<Gtk::Application> app;
        Glib::RefPtr<Gtk::Builder> builder;
        Glib::RefPtr<DrawerType> drawer;
        Glib::RefPtr<Gtk::SpinButton> speed_output;

        void process_thread();
        void on_app_activate();
    public:
        WindowBasic(EngineType &engine);
        ~WindowBasic();
    };

    template<EngineConc E, typename D>
    WindowBasic<E, D>::WindowBasic(E &engine)
        : engine{engine}
    {
        thread = std::thread(std::bind(&WindowBasic::process_thread, this));
    }

    template<EngineConc E, typename D>
    WindowBasic<E, D>::~WindowBasic()
    {
        thread.join();
    }

    template<EngineConc E, typename D>
    void WindowBasic<E, D>::process_thread()
    {
        app = Gtk::Application::create("su.qcrg.gol");
        app->signal_activate().connect(
                std::bind(&ThisType::on_app_activate, this));
        app->signal_shutdown().connect(
                std::bind(&EngineType::quit, &engine));
        if (app->run())
            throw std::runtime_error("GTK: app->run() error");
    }

    template<EngineConc E, typename D>
    void WindowBasic<E, D>::on_app_activate()
    {
        builder = Gtk::Builder::create_from_file(interface_name);
        auto wnd = builder->get_object<Gtk::Window>("window");

        auto main_layout = builder->get_object<Gtk::Box>("main-layout");

        auto draw_widget = builder->get_object<Gtk::DrawingArea>("draw-area");
        drawer = DrawerType::create(engine, draw_widget);
        
        auto action = builder->get_object<Gtk::Button>("action");
        action->set_label(engine.is_played() ? "Pause" : "Play");
        action->signal_clicked().connect([&, action]{
                if (engine.is_played())
                {
                    action->set_label("Play");
                    engine.pause();
                }
                else
                {
                    action->set_label("Pause");
                    engine.play();
                }
            });

        auto one_step = builder->get_object<Gtk::Button>("single-action");
        one_step->signal_clicked().connect([&]{ engine.one_step(); });

        speed_output = builder->get_object<Gtk::SpinButton>("speed");
        speed_output->set_increments(1, 10);
        speed_output->set_range(1u, 10000u);
        speed_output->set_value(engine.get_speed());
        speed_output->signal_value_changed().connect([&]{
                engine.set_speed(speed_output->get_value());
            });

        wnd->show();
        app->add_window(*wnd);
    }

    using Window = WindowBasic<Engine, Drawer>;
} //namespace pnd::gol
