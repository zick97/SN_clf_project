### Supernova Photometric Classification using Machine Learning

For some years now, astrophysics - as well as a wide variety of scientific disciplines - has been facing an era of huge, wide-field surveys that provide access to an ever-increasing amount of data. 
Numerous machine-learning techniques have already been employed: one of the areas in which its use has proved to be of great help is certainly that of __Supernovae__ (SNs).

SNs are of fundamental importance for cosmological studies, in particular for constraining the parameters of the $\Lambda-CDM$ model: for modelling it is first necessary to know the type of event being observed. 
Currently, SN Cosmology is mainly carried out with Type Ia SNs, which have been spectroscopically recognised: this is unfortunately not always possible, and the sample size is therefore limited. 
Recent experiments are able to measure a wider sample, of which, however, only a small portion can be identified spectroscopically. It is therefore required a method that uses __photometric__ measurements to classify the observed SNs.
This project aims to implement such a method, using *supervised* and *unsupervised* learning techniques to make the best use of the available photometric and spectroscopic data, especially the __light curves__ of individual SNSs. 
The dataset consists of tens of thousands of DES simulated light curves: it is publicly available as [Supernova Photometric Classification Challenge](https://www.hep.anl.gov/SNchallenge/) and it is described in [Kessler et al., 2010](https://arxiv.org/abs/1001.5210).
