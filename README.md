# Face-Mask Detection

Environment creation:
1. installing anaconda on your machines
2. go to terminal(mac) or anaconda prompt (Windows) and type: "conda create -n DIP python=3.7"(creates a virtual environemnt)
3. You can activate and deactivate this using: source activate DIP (mac), conda activate DIP (Windows)
4. install the libraries needed by: conda install pandas numpy scipy pillow matplotlib scikit-learn
5. install pytorch based on your specs as: conda install pytorch torchvision torchaudio -c pytorch(installs stable 1.8.1 for mac)
   (https://pytorch.org/) use this link to check according to you
6. Run test.py using python test.py to check if everything runs perfectly.
7. to deactivate run source deactivate (mac) or conda deactivate (Windows)


Everytime you make changes to anything:
git add filename.ext(for all the file names)
git commit -m "(Your message)"
git push

to update the repo in case others have made any changes:
git pull
in case you run into errors (might have to search a bit, Idk what we are supposed to do !!)

Getting the data set off Kagel:
https://www.kaggle.com/andrewmvd/face-mask-detection

go to the data section and download the data set as a zip file. 
U might have to unzip it using something and copy both the directories images and annotation into a new directory "dataset" inside
the DIP-Final repo that you cloned. (do not add the dataset to the actual repository though)
