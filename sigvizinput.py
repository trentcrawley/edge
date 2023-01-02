from tkinter.filedialog import askdirectory

signal = [['vwap','rocvwap1','<',0],['9emaclose1min','rocema9close1min1','<',0]]
daysbefore = 1
daysafter = 1
directory = askdirectory(parent=None, initialdir="/", title='Please select a directory to save charts')
