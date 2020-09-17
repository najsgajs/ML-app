import tkinter as tk

class Start_Page(tk.Frame):
    def __init__(self, parent, controller, ):
        super().__init__(parent, relief="groove", borderwidth=3)

        self.controller = controller
        self.grid_columnconfigure(0, weight=1)
        self.Label = tk.Label(self, text="WITAM i o zdrowie pytam hłe",font=("Times New Roman",25))
        self.Label.grid(row=0,column=0)