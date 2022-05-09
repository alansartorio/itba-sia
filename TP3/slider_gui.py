from typing import Callable, Sequence
import numpy.typing as npt
import numpy as np
import gi
from image_plot import image_evaluate
from activation_functions import get_sigmoid_exp, get_sigmoid_tanh

from network import Network

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
gi.require_version('GdkPixbuf', '2.0')

from gi.repository import GdkPixbuf, Gtk, Adw  # nopep8

FloatArray = npt.NDArray[np.float64]


class LayersSliders(Gtk.Box):
    def __init__(self, weights: Sequence[tuple[int, int]], weight_changed_callback: Callable[[Sequence[FloatArray]], None]) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL)

        def add_layer(layer_weights_shape: tuple[int, int]):
            layer_scroll = Gtk.ScrolledWindow()
            layer_scroll.set_vexpand(True)
            self.append(layer_scroll)

            layer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            layer_scroll.set_child(layer_box)

            def get_slider():
                value = 0
                # ampl = 0.01
                ampl = 0.25
                adj = Gtk.Adjustment(
                    value=value, lower=-ampl, upper=ampl, step_increment=0.001, page_increment=0.01, page_size=0)
                s = Gtk.Scale(adjustment=adj)
                # s.set_vexpand(True)
                s.set_hexpand(True)
                return adj, s

            layer_adjustments = []
            for w in range(np.prod(layer_weights_shape)):
                adj, s = get_slider()
                layer_adjustments.append(adj)
                layer_box.append(s)
            self.adjustments.append(layer_adjustments)

        self.weight_shapes = [layer for layer in weights]
        self.adjustments = []
        for layer in weights:
            add_layer(layer)

        self.connect_sliders()
        self.weight_changed_callback = weight_changed_callback

    def disconnect_sliders(self):
        for adjs, conns in zip(self.adjustments, self.connections):
            for adj, conn in zip(adjs, conns):
                adj.disconnect(conn)
        self.connections.clear()
    def connect_sliders(self):
        self.connections = []
        for adjs in self.adjustments:
            conns = []
            for adj in adjs:
                conn = adj.connect('value-changed', self.weight_changed)
                conns.append(conn)
            self.connections.append(conns)

    @property
    def weights(self):
        weights = []
        for shape, layer in zip(self.weight_shapes, self.adjustments):
            w = np.array([adj.props.value for adj in layer]).reshape(shape)
            weights.append(w)
        return weights

    def weight_changed(self, _):
        self.weight_changed_callback(self.weights)

    def update(self, weights: Sequence[FloatArray]):
        self.disconnect_sliders()
        for layer_weights, adjs in zip(weights, self.adjustments):
            # print(layer_w# eights, layer_weights.flatten(), len(adjs))
            for adj, w in zip(adjs, layer_weights.flatten()):
                adj.props.value = w
        self.connect_sliders()

class MainWindow(Gtk.ApplicationWindow):
    def __init__(self, model: Network, resolution: tuple[int, int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        # scroll = Gtk.ScrolledWindow()
        # self.set_child(scroll)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(box)
        random_button = Gtk.Button(label="Randomize Weights")
        random_button.connect('clicked', self.randomize_weights)
        box.append(random_button)
        self.model = model
        self.image_size = resolution
        # self.pixbuf.scale_simple(200, 200, GdkPixbuf.InterpType.NEAREST)
        self.image = Gtk.Image.new_from_pixbuf(self.get_pixbuf())
        self.image.set_size_request(width=200, height=200)
        # self.image.set_vexpand(True)
        # self.image.set_hexpand(True)
        box.append(self.image)
        self.sliders = LayersSliders([l.weights.shape for l in model.layers], self.update_weights)
        box.append(self.sliders)
        self.update_sliders()

    def get_pixbuf(self):
        im = image_evaluate(self.model, *self.image_size, (-2, 2), (-2, 2))
        # im = im * 255
        im = ((im + 1) / 2) * 255
        im = im.astype(np.uint8)
        w, h, c = im.shape
        assert c == 3
        pb = GdkPixbuf.Pixbuf.new_from_data(
            im.tobytes(), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3)

        return pb

    def update_sliders(self):
        self.sliders.update([l.weights for l in self.model.layers])

    def update_image(self):
        Gtk.Image.set_from_pixbuf(self.image, self.get_pixbuf())

    def randomize_weights(self, _):
        self.model.randomize_weights(0.1)
        self.update_sliders()
        self.update_image()

    def update_weights(self, weights: Sequence[FloatArray]):
        for layer, weights in zip(self.model.layers, weights):
            layer.weights[:, :] = weights
        self.update_image()


class MyApp(Adw.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect('activate', self.on_activate)

    def on_activate(self, app):
        # self.win = MainWindow(Network.with_zeroed_weights(
            # 2, (2, 10, 10, 10, 10, 4, 3), *get_sigmoid_exp(10)), (50, 50), application=app)
        self.win = MainWindow(Network.with_zeroed_weights(
            2, (2, 10, 10, 10, 10, 3), *get_sigmoid_tanh(10)), (50, 50), application=app)
        self.win.present()


app = MyApp(application_id="com.example.GtkSliders")

app.run(None)
