import tkinter as tk
from config import *
from src.utils import face_detection as fd
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk


class FaceDetectionPage(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        self.name = 'FaceDetectionPage'
        self.image = None
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)

        label = tk.Label(self, text="FACE DETECTION", font=LargeFont)
        label.pack(pady=10, padx=10)

        b_oi = tk.Button(self, text="Open image",
                         command=lambda: self.open_image(), width=25, height=1)
        b_oi.pack()

        b_tm = tk.Button(self, text="Template matching",
                         command=lambda: self.template_matching(), width=25, height=1)
        b_tm.pack()

        self.tm_var = tk.StringVar()
        self.tm_var.set('face')
        rb_tm1 = tk.Radiobutton(self, text="Face", value="face", var=self.tm_var)
        rb_tm2 = tk.Radiobutton(self, text="Eyes", value="eyes", var=self.tm_var)
        rb_tm3 = tk.Radiobutton(self, text="Nose&mouse", value="nose&mouse", var=self.tm_var)
        rb_tm4 = tk.Radiobutton(self, text="Eyes&nose", value="eyes&nose", var=self.tm_var)
        rb_tm1.pack()
        rb_tm2.pack()
        rb_tm3.pack()
        rb_tm4.pack()

        b_vj = tk.Button(self, text="Viola Jones",
                         command=lambda: self.viola_jones(), width=25, height=1)
        b_vj.pack()

        self.vj_var = tk.StringVar()
        self.vj_var.set('face&eyes')
        rb_vj1 = tk.Radiobutton(self, text="Face", value="face", var=self.vj_var)
        rb_vj2 = tk.Radiobutton(self, text="Eyes", value="eyes", var=self.vj_var)
        rb_vj3 = tk.Radiobutton(self, text="Face&eyes", value="face&eyes", var=self.vj_var)
        rb_vj1.pack()
        rb_vj2.pack()
        rb_vj3.pack()

        b_sl = tk.Button(self, text="Symmetry lines",
                         command=lambda: self.symmetry_lines(), width=25, height=1)
        b_sl.pack()

        b_back = tk.Button(self, text="Back to Main Menu",
                           command=lambda: controller.show_frame('StartPage'), width=25, height=1)
        b_back.pack()

    def open_image(self):
        img_path = filedialog.askopenfilename()
        if not img_path or img_path.split('.')[-1] not in ['png', 'jpeg', 'jpg']:
            return
        self.image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.draw_image_on_canvas(self.image)

    def template_matching(self):
        if self.image is None:
            return

        result_image = fd.template_matching(self.image, detect=self.tm_var.get())
        self.draw_image_on_canvas(result_image)

    def viola_jones(self):
        if self.image is None:
            return
        result_image = fd.viola_jones(self.image, detect=self.vj_var.get())
        self.draw_image_on_canvas(result_image)

    def symmetry_lines(self):
        if self.image is None:
            return
        face = fd.symmetry_lines(self.image)
        self.draw_image_on_canvas(face)

    def draw_image_on_canvas(self, original_image):
        image = Image.fromarray(original_image).resize((CANVAS_WIDTH, CANVAS_HEIGHT))
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=image)
        self.canvas.pack()
        self.mainloop()
