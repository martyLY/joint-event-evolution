# joint-event-evolution
## Source code for paper "A Joint Model for Event Detection and Evolution"
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
JEE/
├── code
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
│   └── main.py: train model
├── data
│   └── weibo
│       └── CNESC.txt: docuemnts set,the first line is (story_id, event_id ,title, content, keyword, time)
└── README.md
```
