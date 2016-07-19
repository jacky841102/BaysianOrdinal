-------------------------
README File For PeerGrading-Toolkit
-------------------------
Karthik Raman

Version 1.0
04/20/2014

http://peergrading.org


-------------------------
INTRODUCTION
-------------------------

PeerGrading-Toolkit is a python-based toolkit that implements different ordinal peer-grading techniques.


-------------------------
COMPILING
-------------------------

PeerGrading-Toolkit can works in Windows, Linux and Mac environment.

**NOTE**
PeerGrading-Toolkit does require Python version 2.7 or newer in order to run properly.
Additionally certain methods require NumPy and SciPy.

You can download the latest version of Python at http://www.python.org/download/

-------------------------
RUNNING
-------------------------

To run the code: 

Usage:
  peerGrader.py [-h] -i INPUTFILE [-f FORMAT] [-doccol DOCID-COLUMN] [-grcol GRADERID-COLUMN] [-vcol VALUE/GRADE-COLUMN] [-m METHOD] [-iter NUM-ITERATIONS] [--borda] [--kemen] [--all_pairs] [--model_ties] -o OUTPUT-PREFIX -log lOG-FILE [-v VERBOSITY]


Inputs:
  -h, --help            show this help message and exit
  -i INPUTFILE          Input data file (PGF/TSV/CMTXLS format).
  -f FORMAT             Input data format. Options: PGF,TSV,CMTXLS
  -doccol DOCID-COLUMN  Document ID column which contains the ID of the
                        document (index starts from 1). Applicable only for
                        TSV and CMTXLS format files.
  -grcol GRADERID-COLUMN
                        Grader/Reviewer ID column (index starts from 1).
                        Applicable only for TSV and CMTXLS format files.
  -vcol VALUE/GRADE-COLUMN
                        Data value column which contains the grade given
                        (index starts from 1). Applicable only for TSV and
                        CMTXLS format files.
  -m METHOD             Choice of methods to run include MAL (Mallows Model),
                        MALS (Mallows with Scores), BT (Bradley-Terry), THUR
                        (Thurstone Model), PL (Plackett-Luce Model). Also
                        included is the cardinal method: Score-Averaging
                        (SCAVG).
  -iter NUM-ITERATIONS  Number of iterations for estimating reliabilities
  --borda               Use Borda Count for Mallows Model
  --kemen               Use Kemenization for Mallows Model
  --all_pairs           Use All Pairs
  --model_ties          Use variant that models ties
  -o OUTPUT-PREFIX      Output file prefix (two files will be generated: A
                        scores file and a reliabilities file).
  -log LOG-FILE         Log file path.
  -v VERBOSITY          Level of verbosity. Options (in decreasing order):
                        DEBUG/INFO/WARNING/ERROR/CRITICAL
                        
-------------------------
INPUT DATA FORMAT
-------------------------

The function takes in a file as input which can be one of three formats:
a) CMTXLS FORMAT: The file exported from the Microsoft CMT (Conference-Management Toolkit) as an XLS file. If you use this format, you need to provide the column indices of the grader, document and value columns.
b) TSV FORMAT: Tab-separated file format. Like for the CMTXLS format, you need to provide the column indices of the grader, document and value columns.
c) PGF FORMAT: The custom Peer-Grade File Format used by our toolkit. The data from the other formats is converted into this format.

 
-------------------------
PGF DATA FORMAT
-------------------------

The peer-grade file format is simply a succinct line-by-line description of the orderings provided by each grader.

Each line has the following format:
	[TASK-ID] [GRADER-ID] [ORDERING]

The task identifier is is multiple grading tasks are performed and a single grader reliability is desired.
The grader identifier is for identifying the different graders.
 
The ORDERING has the following format:
	[ASSIGNMENT-ID] (Optional-Cardinal-Score) ['>'|'?'] [REMAINING-ORDERING]

The '>' indicates a strict preference.
The '?' represents an unknown preference or no preference.

An example PGF file is given below:
	task1 rvwrid_1 assgnid_1 > assgnid_2 > assgnid_3
	task1 rvwrid_2 assgnid_1 > assgnid_2 > assgnid_3
	task1 rvwrid_3 assgnid_1 > assgnid_3 > assgnid_2

In this example: Reviewer 1 rates assignment 1 as being better than assignment 2 which in turn is better than assignment 3. 
 
Another example with the cardinal score provided:		
	task1 rvwrid_1 assgnid_1 (8.0) > assgnid_2 (7.0) > assgnid_3 (5.0)
	task1 rvwrid_2 assgnid_1 (9.0) > assgnid_2 (7.0) > assgnid_3 (6.0)
	task1 rvwrid_3 assgnid_1 (8.0) > assgnid_3 (6.0) ? assgnid_2 (6.0)
 
Sample files are provided in the package.

-------------------------
OUTPUT DATA FORMAT
-------------------------

Two files are produced as output.

One is the grade file which has the suffix '_docscores.txt'. For each task, this contains the aggregated score for that assignment computed by the method run using the peer grades provided as input. It is of the form:
[ASSIGNMENT-ID] [SCORE]

with the scores sorted in decreasing order.

The second file contains the grader reliabilities (as predicted by the method) and has the suffix '_userrels.txt'. It has the format
[GRADER-ID] [RELIABILITYSCORE]

-------------------------
CONTENTS
-------------------------

The source distribution includes the following files:

1. README.txt : This readme file.
2. LICENSE.txt : License under which software is released.
3. peerGrader.py : The main python interface.

Also included are two directories:
4. sampledata  : Sample PGF data file directory
5. peergrader : Contains the python code


-------------------------
DEBUGGING
-------------------------

In case you have problems with the code you can look for error messages in the log file generated.

If you would like to contact us about bugs/problems with the code please email us at support@peergrading.org


