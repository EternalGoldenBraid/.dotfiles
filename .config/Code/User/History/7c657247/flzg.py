#!/usr/bin/env python
import open3d as o3d
import open3d.visualization.gui as gui

def on_mouse(e):
    if e.type == gui.MouseEvent.Type.BUTTON_DOWN:
        print("[debug] mouse:", (e.x, e.y))
    return gui.Widget.EventCallbackResult.IGNORED

def on_key(e):
    if e.key == gui.KeyName.SPACE:
        if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
            print("[debug] SPACE released")
        else:
            print("[debug] SPACE pressed")
        return gui.Widget.EventCallbackResult.HANDLED
    if e.key == gui.KeyName.W:  # eats W, which is forward in fly mode
        print("[debug] Eating W")
        return gui.Widget.EventCallbackResult.CONSUMED
    return gui.Widget.EventCallbackResult.IGNORED

def main():
    gui.Application.instance.initialize()
    w = gui.Application.instance.create_window("Open3D Example - Events",
                                               640, 480)
    scene = gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
    w.add_child(scene)

    obj = o3d.geometry.TriangleMesh.create_sphere()
    obj.compute_vertex_normals()


    gui.Application.instance.run()

if __name__ == "__main__":
    main()
