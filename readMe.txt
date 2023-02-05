1.Requirement pakage:
    (1)numpy
    (2)torch
    (3)matplotlib
    (4)h5py
    (5)sklearn
    (6)datetime

2.execute with the VScode
3.There are one .py file in the zip : main.py
4.after running the "main.py", it will create (1)folder which save the Loss and accuarcy information of training & validation 
                                                (2)two plots with Loss and accuracy that each plot has 5 subplot(Kfolds = 5)
                                                (3)5 model weights (Kfolds = 5)
5.if you want to run my scratch model, set the varialbe-"resnet_swith" = False, or set = True, before the Kfold Loop.