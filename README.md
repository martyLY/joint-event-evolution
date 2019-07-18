# joint-event-evolution
## Source code for paper "JEDE: Jointly Event Detection and Evolution Model via Siamese GRU Attention Network"
## Requirements
<ol>
  <li>Python 3.6</li>
  <li>numpy</li>
  <li>scipy</li>
  <li>keras 2.2</li>
  <li>My machine with two GPUs (NVIDIA GTX-1080 *2) and two CPUs (Intel Xeon E5-2690 * 2)</li>
</ol>

## Description
```
JEDE/
├── code(1.0)
│   ├── The first version can be seen as a separate implementation of two subtasks.
│   ├── evaluation
│   │   └── evalu.py: evaluate the performance of event cluster and evolution
│   ├── models
│   │   ├── model.py: the model with some functions
│   │   └── similar.py: the simalarity between docuemnts or events
│   ├── preData
│   │   └── data_loader.py: load the data from CNESC.txt 
│   ├── class
│   │   └── my_class.py:custom class including docuemnt,cluster,syory
│   ├── layer
│   │   └── att_layer.py: Attention Mechanism
│   │   └── fir_layer.py: Text preprocessing
│   │   └── cluster_layer.py: documents cluster to events
│   │   └── story_layer.py: events grow to story
│   ├── output
│   │   └── output.py: output to file event_evolution.txt
├── CODE(2.0)
│   ├── The second edition is the final algorithm, improved algorithm and added contrast experiment
│   ├── model.py: the model with some functions
│   ├── similar.py: the simalarity between docuemnts or events
│   ├── data_loader.py: load the data from CNESC.txt 
│   ├── fir_layer.py: Text preprocessing
│   ├── cluster_layer.py: documents cluster to events
│   ├── story_layer.py: events grow to story
│   ├── output.py: output to file event_evolution.txt
│   ├── dbscan.py： contrast algorithm (event detection)
│   ├── lsh.py: contrast algorithm (event detection)
│   ├── jeds.py: contrast algorithm (event detection)
│   └── main.py: train model
├── DATA
│   └── CNESC.txt: docuemnts set,the first line is (story_id, event_id ,title, content, keyword, time)
│   └── our_raw_labled_data: docuemnts set,the first line is (story_id, event_id ,title, content, keyword, time)
└── README.md
```
## Reference 
###  Event evolution/detection comparison algorithm
<ol>
  <li> code : https://github.com/BangLiu/StoryForest.git</li>
  <li> paper: arXiv:1803.00189</li>
  <li> code : CODE/jeds.py</li>
  <li> paper: https://www.ijcai.org/proceedings/2017/581</li>
</ol>
