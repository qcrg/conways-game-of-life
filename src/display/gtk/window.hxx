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
#include <gtkmm/eventcontrollerkey.h>
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
        Glib::RefPtr<Gtk::Button> action_button;
        Glib::RefPtr<Gtk::Button> single_action_button;
        Glib::RefPtr<Gtk::EventControllerKey> key_controller;

        void process_thread();
        void on_app_activate();
        bool process_key_input(guint keyval, guint keysym, Gdk::ModifierType);
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
        
        action_button = builder->get_object<Gtk::Button>("action");
        action_button->set_label(engine.is_played() ? "Pause" : "Play");
        action_button->signal_clicked().connect([&]{
                if (engine.is_played())
                {
                    action_button->set_label("Play");
                    engine.pause();
                }
                else
                {
                    action_button->set_label("Pause");
                    engine.play();
                }
            });

        single_action_button = builder->get_object<Gtk::Button>("single-action");
        single_action_button->signal_clicked().connect([&]{ engine.one_step(); });

        speed_output = builder->get_object<Gtk::SpinButton>("speed");
        speed_output->set_increments(1, 10);
        speed_output->set_range(1u, 10000u);
        speed_output->set_value(engine.get_speed());
        speed_output->signal_value_changed().connect([&]{
                engine.set_speed(speed_output->get_value());
            });

        key_controller = Gtk::EventControllerKey::create();
        wnd->add_controller(key_controller);
        key_controller->signal_key_pressed().connect(
                sigc::mem_fun(*this, &ThisType::process_key_input), true);

        wnd->show();
        app->add_window(*wnd);
    }

    template<EngineConc E, typename D>
    bool WindowBasic<E, D>::process_key_input(guint keyval, guint, Gdk::ModifierType)
    {
        auto change_speed = [&](Gtk::SpinType spin_type) {
            double incr, page_incr;
            speed_output->get_increments(incr, page_incr);
            speed_output->spin(spin_type, incr);
        };
        switch (keyval)
        {
            case GDK_KEY_h:
            case GDK_KEY_H:
            case GDK_KEY_leftarrow:
                drawer->left();
            break;

            case GDK_KEY_l:
            case GDK_KEY_L:
            case GDK_KEY_rightarrow:
                drawer->right();
            break;

            case GDK_KEY_j:
            case GDK_KEY_J:
            case GDK_KEY_downarrow:
                drawer->down();
            break;

            case GDK_KEY_k:
            case GDK_KEY_K:
            case GDK_KEY_uparrow:
                drawer->up();
            break;

            case GDK_KEY_minus:
            case GDK_KEY_underscore:
                change_speed(Gtk::SpinType::STEP_BACKWARD);
            break;

            case GDK_KEY_plus:
            case GDK_KEY_equal:
                change_speed(Gtk::SpinType::STEP_FORWARD);
            break;

            case GDK_KEY_p:
            case GDK_KEY_P:
                g_signal_emit_by_name(action_button->gobj(), "clicked");
            break;
            
            case GDK_KEY_o:
            case GDK_KEY_O:
                g_signal_emit_by_name(single_action_button->gobj(), "clicked");
            break;
        }
        return true;
    }

    using Window = WindowBasic<Engine, Drawer>;
} //namespace pnd::gol
