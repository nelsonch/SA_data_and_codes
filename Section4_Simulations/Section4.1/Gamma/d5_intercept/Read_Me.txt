The run.sh file is the shell script that will kick off the runs in background on a linux server. Please make sure to properly change the working directory and the executable python PATH. Please do run the codes on a server since local laptop might not have the enough computing power to finish the runs in a reasonable amount of time. 

It will write out a csv file. Open that csv file and the first $d$ columns contains the estimated fixed effects as used and reported in the paper. 

The computing environment I used are as follows.

Amazon Linux 
Intel Xeon Platinum 8259CL 32 CPUs
Python version: 3.5.3
numpy version: 1.16.4
scipy version: 1.3.0
pandas version: 0.20.1