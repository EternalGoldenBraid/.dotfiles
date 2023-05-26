import open3d as o3d
from open3d.visualization import gui, rendering
from open3d.visualization import O3DVisualizer

def draw(geometry=None,
         title="Open3D",
         width=1024,
         height=768,
         actions=None,
         lookat=None,
         eye=None,
         up=None,
         field_of_view=60.0,
         bg_color=(1.0, 1.0, 1.0, 1.0),
         bg_image=None,
         show_ui=None,
         point_size=None,
         line_width=None,
         animation_time_step=1.0,
         animation_duration=None,
         on_init=None,
         on_animation_frame=None,
         on_animation_tick=None,
         non_blocking_and_return_uid=False):
    gui.Application.instance.initialize()
    w = O3DVisualizer(title, width, height)
    w.set_background(bg_color, bg_image)

    if actions is not None:
        for a in actions:
            w.add_action(a[0], a[1])

    if point_size is not None:
        w.point_size = point_size

    if line_width is not None:
        w.line_width = line_width
        
    def add(g, n):
        if isinstance(g, dict):
            w.add_geometry(g)
        else:
            w.add_geometry("Object " + str(n), g)

    n = 1
    if isinstance(geometry, list):
        for g in geometry:
            add(g, n)
            n += 1
    elif geometry is not None:
        add(geometry, n)

    w.reset_camera_to_default()  # make sure far/near get setup nicely
    if lookat is not None and eye is not None and up is not None:
        w.setup_camera(field_of_view, lookat, eye, up)

    w.animation_time_step = animation_time_step
    if animation_duration is not None:
        w.animation_duration = animation_duration

    if show_ui is not None:
        w.show_settings = show_ui

    if on_init is not None:
        on_init(w)
    if on_animation_frame is not None:
        w.set_on_animation_frame(on_animation_frame)
    if on_animation_tick is not None:
        w.set_on_animation_tick(on_animation_tick)
        
        
    ### Sliders for debug
    slider = gui.Slider(gui.Slider.INT)
    em = 10
    slider.set_limits(5, 13)
    collapse = gui.CollapsableVert("Widgets", 0.33 * em,
        gui.Margins(em, 0, 0, 0))
    # slider.set_on_value_changed(self._on_slider)
    # collapse.add_child(slider)

    gui.Application.instance.add_window(w)
    if non_blocking_and_return_uid:
        return w.uid
    else:
        gui.Application.instance.run()
