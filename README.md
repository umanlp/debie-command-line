# DEBIE Command-line Tool

CLI Tool for the web application DEBIE.  
This tool can evaluate bias specifications and debias embedding spaces based on explicit bias specifications.  
The tool features following evaluation scores: ECT, BAM, WEAT, K-Means++, SVM, SimLex and WordSim.  
The following debiasing models are included: BAM, GBDD, BAM x GBDD and GBDD x BAM.  
For further detail to the used scores and models, please visit the information pages of the web application DEBIE.

### Installation & Starting
For setting up the CLI version of debie, download the GitHub-Code and run following commands:  
&nbsp;&nbsp;&nbsp;&nbsp;- pip install -r requirements.txt  
&nbsp;&nbsp;&nbsp;&nbsp;- python debie.py  


### Direct Calls
For executing direct evaluation or debiasing commands, please provide the required information in a json-file.  
An example how this file should look like can be found as exec-file.json here on GitHub.  
For bias evaluation run following command:  
&nbsp;&nbsp;&nbsp;&nbsp;- python evaluation.py path/to/exec-file.json  
For debiasing run following command:  
&nbsp;&nbsp;&nbsp;&nbsp;- pthon debiasing.py path/to/exec-file.json
