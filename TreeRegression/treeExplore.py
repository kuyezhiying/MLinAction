from numpy import *
from Tkinter import *
import cart
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if checkBtnVar.get() :
        if tolN < 2 :
            tolN = 2
        modelTree = cart.createTree(reDraw.rawData, cart.modelLeaf, cart.modelError, (tolS, tolN))
        yHat = cart.createForecast(modelTree, reDraw.testData, cart.modelTreeEval)
    else :
        regTree = cart.createTree(reDraw.rawData, ops = (tolS, tolN))
        yHat = cart.createForecast(regTree, reDraw.testData)
    reDraw.a.scatter(reDraw.rawData[:, 0].A1, reDraw.rawData[:, 1].A1, s = 5)
    reDraw.a.plot(reDraw.testData, yHat, linewidth = 2.0)
    reDraw.canvas.show()

def getInputs():
    try :
        tolN = int(tolNEntry.get())
    except :
        tolN = 10
        print "Please enter an integer for tolN"
        tolNEntry.delete(0, END)
        tolNEntry.insert(0, '10')
    try :
        tolS = float(tolSEntry.get())
    except :
        tolS = 1.0
        print "Please enter a float for tolS"
        tolSEntry.delete(0, END)
        tolSEntry.insert(0, '1.0')
    return tolN, tolS

def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)

root = Tk()

#Label(root, text = "Plot Place Holder").grid(row = 0, columnspan = 3)

reDraw.f = Figure(figsize = (5, 4), dpi = 100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master = root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row = 0, columnspan = 3)

Label(root, text = "tolN").grid(row  = 1, column = 0)
tolNEntry = Entry(root)
tolNEntry.grid(row = 1, column = 1)
tolNEntry.insert(0, '10')

Label(root, text = "tolS").grid(row  = 2, column = 0)
tolSEntry = Entry(root)
tolSEntry.grid(row = 2, column = 1)
tolSEntry.insert(0, '1.0') 
Button(root, text = "ReDraw", command = drawNewTree).grid(row = 1, column = 2, rowspan = 3)

checkBtnVar = IntVar()
checkBtn = Checkbutton(root, text = "Model Tree", variable = checkBtnVar)
checkBtn.grid(row = 3, column = 0, columnspan = 2)

reDraw.rawData = mat(cart.loadDataSet('sine.txt'))
reDraw.testData = arange(min(reDraw.rawData[:, 0]), max(reDraw.rawData[:, 0]), 0.01)

reDraw(1.0, 10)

root.mainloop()


