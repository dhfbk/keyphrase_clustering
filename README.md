# keyphrase_clustering


This script takes as input a folder containing one or more lists of keyphrases and return the keyphrases organized into clusters according to their similarity.

To obtain the list of keywods we relied on [KD: Keyphrase Digger](http://dh.fbk.eu/technologies/kd).

An example of the input file is available in `sample_input/sample.txt`. Each line should contain a keyphrase and an associated weight (teb separated). Keyphrases made of two or more words need to be joint by an underscore.

To use the code, unzip the file `glove.6B.50d.txt.zip` in the main folder and run:
```
python keyphrase_aggregator.py -d sample_input/
```

Parameters:
* `-d`  directory containing the list of keyphrases
* `-v`  to use a vectors file different from the file `glove.6B.50d.txt` available in the folder
* `-t`  to change the cosine similarity threshold used for building the graph (default is 0.8). Using low values can result vn very long processing time

In addition, if you want to obtain cluster with a maximum length you can set the parameters:
* `-m`  maximum number of keyphrase in clusters
* `-s`  cosine similarity threshold used to split the large cluster (if set should be a value higher than `-t`)

The code is tested on macOS and linux.
