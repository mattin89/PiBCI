# PiBCI
Author: Mario De Lorenzo
Email: mario.de_lorenzo@tu-dresden

Opensource GUI to connect any Bluetooth/WiFi board to a Raspberry Pi4. The goal is to make BCI projects accessible and easy to do for everybody and for any budgets.

The available portable BCI devices can be expensive and hard to use especially for people with no experience and training. More advanced electroencephalography can even cost tens or hundreds of thousands of dollars. This project shows it is possible to conduct neuroscience experiments and to build modern portable BCI devices while still having high performances and make it accessible to everybody.

![Alt text](https://github.com/mattin89/PiBCI/blob/338e26e2cfbadc4464abcb4667b95d9f686974cf/Untitled_design-removebg-preview%20(1).png "Logo")

The main functions of the GUI are:
- **Plotting:** it plots in real time from 8 channels
- **Motor Imagery Tasks:** with this function, everybody can do a simple motor imagery experiment. Just as I explained last post, the function will show you a left and a right arrow one at a time. During that time, the subject will think of a movement (like opening and closing right or left hand) so to get data of motor imagery tasks from each individual. The experiment is completely customizable. You can input the display time, resting time, how many sessions, and where you want to save the data. At the end of the experiment, the GUI will save a csv file with 8 channels plus labels for each session. In future, I might add the function the user can choose his/her own pictures and also to add more than two classes.
- **Machine Learning:** you can read the csv file that it was generated and perform basic feature extraction (filtering data from 8 to 30Hz) and some signal processing (I used common spatial pattern since it is a common filtering algorithm for this task). Finally, it uses linear SVM to classify the data, print the accuracy and save the model. On windows, I managed to run also CNN layers but I am having troubles installing tensorflow on the Raspberry Pi4. If somebody knows some easy solution, please let me know. I managed to install everything else like pandas, sklearn, numpy, etc.


The next steps for the GUI are:
- Make the GUI work also for high sampling rate
- Include more machine learning algorithms and more choices for the data cleaning and feature extraction
- Include more neuroscience experiments such as P300 and SSVEP
- Make the GUI look prettier
- Get more people involved with PiBCI so to create a community of builders

If you want to learn more about how the GUI started and how I connected my Huzzah32 board via Bluetooth and UDP, you can find my posts here:
https://community.element14.com/challenges-projects/design-challenges/sci-pi-design-challenge/?ICID=sci-pi-DCH
