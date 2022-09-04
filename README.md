# Federated-Learning-using-Cloudlets
This code repository uses a modified version of the FedProx repository (you can find it inside the FedProx folder) to support the idea of off-loading local training of Machine Learning models from edge device to cloudlets, which are mini-cloud networks.

## Federated Learning -- a brief introduction
The following serves as a very higher level overview of Federated Learning. For a better overview, you can visit [this Federated Learning introductory blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html).

Federated Learning is a process that allows model training on user data without the user's data being stored in a central database. This is made possible by having devices locally train Machine Learning models on their end and then sending back the updates from their trainig to a central server that averages/aggregates model updates from all devices to create a global model. 

## The problem we try to solve using cloudlets
One major problem with the standard Federated Learning algorithm is that devices with low-end computational (example: smaller number of CPU cores, low RAM) resources and/or network resources (example: low internet bandwidth) are unable to perform their local model training and report back to the aggregating server within the alloted round time. This project aims to solve this problem by offloading some of this computation to cloudlets.

By offloading computation to cloudlets, we directly solve the systems heterogenity problem sicne if each device's local learning happens on a virtual machine on a cloudlet, we can be sure that each device has roughly equal computational resources alloted to the model training process. To ensure network variability, we propose a series of network profiling techniques that make use of statistics to place bounds of network delays to efficiently minimize device dropouts (i.e: devices unable to complete model training within their alloted time).

This solution does require data to leave user's device and go to a cloudlet. However, using privacy policy enforcement mediators, users can decide whether or not they wish to delete their data after the training process is over.

A more thorough outline of this solution can be found at [this link](https://github.com/mahad852/Federated-Learning-using-Cloudlets/blob/master/Outline.pdf). In case you wish to reed a detailed discussion of the solution alongside experimental results from the simulations that we performed, please refer to the "Main Findings" section below.

## Steps to run
Steps to run are similar to the steps performed when running FedProx experiments, and as such you should follow the ReadME in the FedProx project to reproduce some of the graph plots attached in this repositiory.

## Main findings
The main findings of this project can be found at [this link](https://drive.google.com/file/d/1Hb5ZXaVDKM4ySLckQeQ67xD1gfwmVg1g/view?usp=sharing).
