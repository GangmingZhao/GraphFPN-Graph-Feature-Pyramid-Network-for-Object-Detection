# GraphFPN: Graph Feature Pyramid Network for Object Detection

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
├── src     
├── README.md 
└── requirements.txt
```
