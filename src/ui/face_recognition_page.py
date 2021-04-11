import tkinter as tk
from src.ui.config import *
from src.utils.face_recognition import *
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from PIL import Image, ImageTk


class FaceRecognitionPage(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        self.parent = parent
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)

        label = tk.Label(self, text="FACE RECOGNITION", font=LargeFont)
        label.pack(pady=10, padx=10)
        label_choose_method = tk.Label(self, text="Choose method:", font=LargeFont)
        label_choose_method.pack(pady=10, padx=10)

        self.method_var = tk.StringVar()
        self.method_var.set('histogram')
        rb_method1 = tk.Radiobutton(self, text="Histogram", value="histogram", var=self.method_var)
        rb_method2 = tk.Radiobutton(self, text="Dft", value="dft", var=self.method_var)
        rb_method3 = tk.Radiobutton(self, text="Dct", value="dct", var=self.method_var)
        rb_method4 = tk.Radiobutton(self, text="Gradient", value="gradient", var=self.method_var)
        rb_method5 = tk.Radiobutton(self, text="Scale", value="scale", var=self.method_var)
        rb_method1.pack()
        rb_method2.pack()
        rb_method3.pack()
        rb_method4.pack()
        rb_method5.pack()

        label_drawing = tk.Label(self, text='Lets draw a graph of dependence:', font=LargeFont, pady=25)
        label_drawing.pack()

        self.graph_var = tk.StringVar()
        self.graph_var.set('param')
        rb_graph1 = tk.Radiobutton(self, text="Param", value="param", var=self.graph_var)
        rb_graph2 = tk.Radiobutton(self, text="Size", value="size", var=self.graph_var)
        rb_graph1.pack()
        rb_graph2.pack()

        label_size = tk.Label(self, text='Fixed variable:', font=LargeFont)
        label_size.pack()
        self.e_size = tk.Entry(self)
        self.e_size.insert(tk.END, '5')
        self.e_size.pack()

        self.canvas = tk.Canvas(self, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)
        self.canvas.pack()

        b_graph = tk.Button(self, text='Draw',
                            command=lambda: self.draw_graph(), width=25, height=1)
        b_graph.pack()

        b_best_params = tk.Button(self, text="Calculate best params",
                                  command=lambda: self.calc_best_params(), width=25, height=1)
        b_best_params.pack()

        self.l_best_params = tk.Label(self, text='', font=LargeFont)
        self.l_best_params.pack(pady=10, padx=10)

        b_back = tk.Button(self, text="Back to Main Menu",
                           command=lambda: controller.show_frame('StartPage'), width=25, height=1)
        b_back.pack()

    def calc_best_params(self):
        self.l_best_params.configure(text='Testing...')
        self.parent.update()
        method = self.method_var.get()
        params = {
            'histogram': {
                'start': 2,
                'stop': 10,  # 64
                'step': 4, },
            'dft': {
                'start': 20,
                'stop': 64,
                'step': 2, },
            'dct': {
                'start': 20,
                'stop': 64,
                'step': 2, },
            'scale': {
                'start': 2,
                'stop': 16,
                'step': 2, },
            'gradient': {
                'start': 2,
                'stop': 60,
                'step': 2, },
        }
        results = search_best_param(method, **params[method])
        best_params = results['best_params']
        param = best_params['param']
        size = best_params['size']
        score = results['best_score']
        self.l_best_params.configure(text='Best score: {:.2f}% on param:'
                                          ' {} with train size per class: {}'.format(100 * score, param, size))

    def draw_graph(self):
        method = self.method_var.get()
        tmp_path = Path().cwd().joinpath('data/tmp').joinpath("tmp_graph.png")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        param = int(self.e_size.get())
        if self.graph_var.get() == 'size':
            x, y = get_size_depend(method, param)
            ax.set(xlabel='train size', ylabel='accuracy (%)',
                   title='Size dependence')
        else:

            x, y = get_param_depend(method, param, params_grid[method])
            ax.set(xlabel='parameter', ylabel='accuracy (%)',
                   title='Parameter dependence')

        ax.plot(x, 100*y)
        ax.grid()
        fig.savefig(tmp_path)
        image = cv2.imread(str(tmp_path))
        self.draw_graph_on_canvas(image)

    def draw_graph_on_canvas(self, original_image):
        image = Image.fromarray(original_image).resize((GRAPH_WIDTH, GRAPH_HEIGHT))
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=image)
        self.canvas.pack()
        self.mainloop()
