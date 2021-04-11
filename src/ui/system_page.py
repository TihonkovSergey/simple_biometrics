import tkinter as tk
from src.ui.config import *
from src.utils.face_recognition import *
from src.utils.face_recognition_system import FaceClassifierSystem
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog


class FaceRecognitionSystemPage(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        self.parent = parent
        super().__init__(parent, *args, **kwargs)
        self.image = None

        label = tk.Label(self, text="FACE RECOGNITION SYSTEM", font=LargeFont)
        label.pack(pady=10, padx=10)

        label_size = tk.Label(self, text='Train size per class:', font=LargeFont)
        label_size.pack()
        self.e_size = tk.Entry(self)
        self.e_size.insert(tk.END, '5')
        self.e_size.pack()

        b_best_params = tk.Button(self, text="Calculate best params",
                                  command=lambda: self.calc_best_system_params(), width=25, height=1)
        b_best_params.pack()

        self.l_best_params = tk.Label(self, text='', font=LargeFont)
        self.l_best_params.pack(pady=10, padx=10)

        b_oi = tk.Button(self, text="Open image",
                         command=lambda: self.open_image(), width=25, height=1)
        b_oi.pack()

        self.canvas = tk.Canvas(self, width=GRAPH_WIDTH, height=GRAPH_HEIGHT)
        self.canvas.pack()

        b_pred = tk.Button(self, text="Predict class",
                           command=lambda: self.predict_image_class(), width=25, height=1)
        b_pred.pack()

        self.label_class = tk.Label(self, text='CLASS: ?', font=LargeFont)
        self.label_class.pack(pady=10, padx=10)

        b_back = tk.Button(self, text="Back to Main Menu",
                           command=lambda: controller.show_frame('StartPage'), width=25, height=1)
        b_back.pack()

    def calc_best_system_params(self):
        size = int(self.e_size.get())
        best_params = {}
        for method, param_list in params_grid.items():
            p, v = get_param_depend(method, size, param_list)
            best_param, best_score = None, 0
            for pp, vv in zip(p, v):
                if vv > best_score:
                    best_param = pp
                    best_score = vv
            best_params[method] = best_param
        msg = "\n".join("{}: {}".format(method, value) for method, value in best_params.items())
        self.l_best_params.configure(text=msg)

    def open_image(self):
        img_path = filedialog.askopenfilename()
        if not img_path or img_path.split('.')[-1] not in ['png', 'jpeg', 'jpg']:
            return
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.draw_image_on_canvas(self.image)

    def draw_image_on_canvas(self, original_image):
        image = Image.fromarray(original_image).resize((GRAPH_WIDTH, GRAPH_HEIGHT))
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=image)
        self.mainloop()

    def predict_image_class(self):
        if self.image is None:
            return
        clf = FaceClassifierSystem(method_parameters='best')
        data, labels = get_dataset()
        clf.fit(data, labels)
        label = clf.predict([self.image])
        self.label_class.configure(text="CLASS: {}".format(label))
