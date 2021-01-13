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
An example how this file should look like is uploaded as set8_config_example.json here on GitHub.  
For bias evaluation run following command:  
&nbsp;&nbsp;&nbsp;&nbsp; python debie.py --mode=evaluation --config=path/to/config-file.json  
For debiasing run following command:  
&nbsp;&nbsp;&nbsp;&nbsp; python debie.py --mode=debiasing --config=path/to/config-file.json  


### Impressum
This tool is also available as a web-application under: http://wifo5-29.informatik.uni-mannheim.de/.  
The included models and methods are based on the paper "A General Framework for Implicit and Explicit Debiasing of Distributional Word Vector Spaces" by Lauscher et al.,
available under https://arxiv.org/abs/1909.06092.  
This tool has been developed by Niklas Friedrich and Anne Lauscher.
