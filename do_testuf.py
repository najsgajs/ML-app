import tkinter as tk



class main(tk.Frame):
    def __init__(self,parent):
        super().__init__(parent)
        self.old = ""
        self.execution()
        self.text()

    def execution(self):
        self.sv = tk.StringVar()
        self.sv.trace('w', self.callback)
        e = tk.Entry(self, textvariable=self.sv)
        e.pack()

    def text(self):
        self.txt = tk.Text()
        self.txt.insert("1.0","Aawfawfawdawd\nawdawfawfaw\n")
        self.txt.pack()
        self.txt.tag_configure("highlight", background="red")
        self.txt.tag_configure("normal", background="yellow")



    def callback(self,*args):
        if self.sv.get() == '':
            print(self.sv.get())
            self.txt.tag_remove("highlight", "{}.0".format(int(self.old)), "{}.end+1c".format(self.old))
            print('xd')
        else:
            print(self.sv.get())
            self.txt.tag_add("highlight", "{}.0".format(int(self.sv.get())), "{}.end+1c".format(self.sv.get()))
            self.old = self.sv.get()



root = tk.Tk()
main(root).pack()
root.geometry("480x200")
root.mainloop()