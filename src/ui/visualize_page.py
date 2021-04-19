import tkinter as tk
from src.ui.config import *
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import numpy as np
from src.utils.visualization_support import VisualizeConnector
import matplotlib.pyplot as plt


class VisualizeTestPage(tk.Tk):
    def __init__(self, controller=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = controller
        self.images = {}
        self.canvas = {}
        self.var = {}
        self.image_id = {}
        self.methods_list = ['system', 'histogram', 'dft', 'dct', 'scale', 'gradient']

        self.geometry('{}x{}'.format(VISUALIZE_TEST_PAGE_WIDTH, VISUALIZE_TEST_PAGE_HEIGHT))

        self.l_size = tk.Label(self, text='train size per class', font=LargeFont)
        self.l_size.grid(row=0, column=0)

        self.sp_size = tk.Spinbox(width=7, from_=1, to=9)
        self.sp_size.grid(row=1, column=0)

        self.var['histogram'] = tk.BooleanVar()
        self.var['histogram'].set(1)
        cb_histogram = tk.Checkbutton(self, text="Histogram",
                                      variable=self.var['histogram'],
                                      onvalue=1, offvalue=0)
        cb_histogram.grid(row=2, column=0)

        self.var['dft'] = tk.BooleanVar()
        self.var['dft'].set(1)
        cb_dft = tk.Checkbutton(self, text="DFT",
                                variable=self.var['dft'],
                                onvalue=1, offvalue=0)
        cb_dft.grid(row=3, column=0)

        self.var['dct'] = tk.BooleanVar()
        self.var['dct'].set(1)
        cb_dct = tk.Checkbutton(self, text="DCT",
                                variable=self.var['dct'],
                                onvalue=1, offvalue=0)
        cb_dct.grid(row=3, column=0)

        self.var['scale'] = tk.BooleanVar()
        self.var['scale'].set(1)
        cb_scale = tk.Checkbutton(self, text="Scale",
                                  variable=self.var['scale'],
                                  onvalue=1, offvalue=0)
        cb_scale.grid(row=4, column=0)

        self.var['gradient'] = tk.BooleanVar()
        self.var['gradient'].set(1)
        cb_gradient = tk.Checkbutton(self, text="Gradient",
                                     variable=self.var['gradient'],
                                     onvalue=1, offvalue=0)
        cb_gradient.grid(row=5, column=0)

        self.var['system'] = tk.BooleanVar()
        self.var['system'].set(1)
        cb_system = tk.Checkbutton(self, text="System",
                                   variable=self.var['system'],
                                   onvalue=1, offvalue=0)
        cb_system.grid(row=6, column=0)

        # Test image
        test_name = tk.Label(self, text='TEST IMAGE & System prediction', font=LargeFont)
        test_name.grid(row=0, column=1)
        self.canvas['system'] = tk.Canvas(self, width=VISUALIZE_TEST_PAGE_IMAGE_WIDTH,
                                          height=VISUALIZE_TEST_PAGE_IMAGE_HEIGHT)
        self.image_id['system'] = self.canvas['system'].create_image(0, 0, anchor="nw")
        self.canvas['system'].grid(row=1, column=1, rowspan=2)

        # DFT image
        l_dft_name = tk.Label(self, text='DFT', font=LargeFont)
        l_dft_name.grid(row=0, column=2)
        self.canvas['dft'] = tk.Canvas(self, width=VISUALIZE_TEST_PAGE_IMAGE_WIDTH,
                                       height=VISUALIZE_TEST_PAGE_IMAGE_HEIGHT)
        self.image_id['dft'] = self.canvas['dft'].create_image(0, 0, anchor="nw")
        self.canvas['dft'].grid(row=1, column=2, rowspan=2)

        # DCT image
        l_dct_name = tk.Label(self, text='DCT', font=LargeFont)
        l_dct_name.grid(row=0, column=3)
        self.canvas['dct'] = tk.Canvas(self, width=VISUALIZE_TEST_PAGE_IMAGE_WIDTH,
                                       height=VISUALIZE_TEST_PAGE_IMAGE_HEIGHT)
        self.image_id['dct'] = self.canvas['dct'].create_image(0, 0, anchor="nw")
        self.canvas['dct'].grid(row=1, column=3, rowspan=2)

        # Histogram image
        l_histogram_name = tk.Label(self, text='Histogram', font=LargeFont)
        l_histogram_name.grid(row=3, column=1)
        self.canvas['histogram'] = tk.Canvas(self, width=VISUALIZE_TEST_PAGE_IMAGE_WIDTH,
                                             height=VISUALIZE_TEST_PAGE_IMAGE_HEIGHT)
        self.image_id['histogram'] = self.canvas['histogram'].create_image(0, 0, anchor="nw")
        self.canvas['histogram'].grid(row=4, column=1, rowspan=2)

        # Scale image
        l_scale_name = tk.Label(self, text='Scale', font=LargeFont)
        l_scale_name.grid(row=3, column=2)
        self.canvas['scale'] = tk.Canvas(self, width=VISUALIZE_TEST_PAGE_IMAGE_WIDTH,
                                         height=VISUALIZE_TEST_PAGE_IMAGE_HEIGHT)
        self.image_id['scale'] = self.canvas['scale'].create_image(0, 0, anchor="nw")
        self.canvas['scale'].grid(row=4, column=2, rowspan=2)

        # Gradient image
        l_gradient_name = tk.Label(self, text='Gradient', font=LargeFont)
        l_gradient_name.grid(row=3, column=3)
        self.canvas['gradient'] = tk.Canvas(self, width=VISUALIZE_TEST_PAGE_IMAGE_WIDTH,
                                            height=VISUALIZE_TEST_PAGE_IMAGE_HEIGHT)
        self.image_id['gradient'] = self.canvas['gradient'].create_image(0, 0, anchor="nw")
        self.canvas['gradient'].grid(row=4, column=3, rowspan=2)

        # Graph
        self.canvas['graph'] = tk.Canvas(self, width=VISUALIZE_TEST_PAGE_GRAPH_WIDTH,
                                         height=VISUALIZE_TEST_PAGE_GRAPH_HEIGHT)
        self.image_id['graph'] = self.canvas['graph'].create_image(0, 0, anchor="nw")
        self.canvas['graph'].grid(row=6, column=1, rowspan=3, columnspan=3)

        # Buttons
        b_prev = tk.Button(self, text="Previous",
                           command=lambda: self._prev_slide(), width=20, height=1)
        b_prev.grid(row=9, column=0)

        b_start = tk.Button(self, text="Start/Stop",
                            command=lambda: self.start(), width=20, height=1)
        b_start.grid(row=9, column=1)

        b_next = tk.Button(self, text="Next",
                           command=lambda: self._next_slide(), width=20, height=1)
        b_next.grid(row=9, column=2)

        b_main_menu = tk.Button(self, text="Back to Main Menu",
                                command=lambda: self._main_menu(), width=20, height=1)
        b_main_menu.grid(row=9, column=3)

        self.is_run = False
        self.vis_connector = None

        for key in self.canvas:
            self._set_black(key)

    def _main_menu(self):
        if self.controller is None:
            return
        self.destroy()
        app = self.controller.__class__()
        app.mainloop()

    def _update_image(self, image, method, frame_color=None):
        if method == 'graph':
            w, h = VISUALIZE_TEST_PAGE_GRAPH_WIDTH, VISUALIZE_TEST_PAGE_GRAPH_HEIGHT
        else:
            w, h = VISUALIZE_TEST_PAGE_IMAGE_WIDTH, VISUALIZE_TEST_PAGE_IMAGE_HEIGHT

        image = Image.fromarray(image).resize((w, h))
        image = ImageTk.PhotoImage(image)
        self.images[method] = image
        self.canvas[method].itemconfig(self.image_id[method], image=self.images[method])
        if frame_color is not None:
            self.canvas[method].create_rectangle(w, 0, 0, h, outline=frame_color, width=4)
        self.update()

    def _update_many_images(self, results):
        methods_to_draw = self._get_methods_to_draw()
        methods_to_draw['system'] = True

        for method, need_to_draw in methods_to_draw.items():
            if need_to_draw:
                res = results[method]
                image = res['image']
                if res['is_correct']:
                    color = '#0f0'  # green
                else:
                    color = '#f00'  # red
                self._update_image(image, method, frame_color=color)
            else:
                self._set_black(method)

    def _prev_slide(self):
        self._check_vis_connector()
        results, lines = self.vis_connector.get_prev()
        if results is not None:
            self._update_many_images(results)
            self._plot_lines(lines)

    def _next_slide(self):
        self._check_vis_connector()
        results, lines = self.vis_connector.get_next()
        if results is not None:
            self._update_many_images(results)
            self._plot_lines(lines)

    def _repeater(self):
        if not self.is_run:
            return
        self._next_slide()
        self.after(4000, self._repeater)

    def start(self):
        self.is_run = not self.is_run
        if not self.is_run:
            return
        self._check_vis_connector()
        self._repeater()

    def _get_methods_to_draw(self):
        return {method: self.var[method].get() for method in self.methods_list}

    def _set_black(self, method):
        path = Path().cwd().joinpath('data/x_image.png')
        image = cv2.imread(str(path))
        self._update_image(image, method)

    def _check_vis_connector(self):
        if self.vis_connector is None:
            self.vis_connector = VisualizeConnector(size=int(self.sp_size.get()))

    def _plot_lines(self, lines):
        tmp_path = Path().cwd().joinpath('data/tmp/tmp_graph.png')
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        ax.set(xlabel='count', ylabel='precision (%)', title='Real-time test precision')

        methods_to_draw = self._get_methods_to_draw()
        for method in methods_to_draw:
            if methods_to_draw[method]:
                line = lines[method]
                ax.plot(range(len(line)), line, label=method)
        ax.grid()
        ax.legend()
        fig.savefig(tmp_path)
        image = cv2.imread(str(tmp_path))
        self._update_image(image, 'graph')
