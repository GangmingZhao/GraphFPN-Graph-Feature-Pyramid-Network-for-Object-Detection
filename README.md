# GraphFPN: Graph Feature Pyramid Network for Object Detection with Retinanet
Inspired by Graph_FPN and RetinaNet, we have used the Graph_FPN structure as a backbone to train on Retinanet and the work is not yet complete.  
https://arxiv.org/abs/2108.00580   


For demonstraction with Graph_fpn, run:
~~~
python demo.py
~~~

For demonstraction with fpn, run:
~~~
python demo.py --no_graph
~~~

For training , run:
~~~
python train.py
~~~

For test with Graph_fpn, run
~~~
python test.py
~~~

For test with fpn, run
~~~
python test.py --no_graph
~~~

## Folder structure

```
${ROOT}
└── checkpoint/
└── COCO/    
│   └── coco/
│   │    ├── .config 
│   │    ├── 2017/
│   │
│   ├── downloads/
│
│
└── data_demo/
|   ├── data
|   |    ├── coco
|   |    ├── checkpoint
|   ├── data.zip
|
├── results
├── src     
├── README.md 
└── requirements.txt
```

## References
[1] Retinanet: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) <br>
