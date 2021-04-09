import tkinter as tk
from config import *


class StartPage(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        self.name = 'StartPage'
        super().__init__(parent, *args, **kwargs)
        label = tk.Label(self, text="MAIN MENU", font=LargeFont)
        label.pack(pady=10, padx=10)

        button1 = tk.Button(self, text="Face Detection",
                            command=lambda: controller.show_frame('FaceDetectionPage'), width=20, height=1)
        button2 = tk.Button(self, text=" Project2 ",
                            command=lambda: controller.show_frame('FaceDetectionPage'), width=20, height=1)
        button3 = tk.Button(self, text="   Project3   ",
                            command=lambda: controller.show_frame('FaceDetectionPage'), width=20, height=1)

        button1.pack()
        button2.pack()
        button3.pack()
