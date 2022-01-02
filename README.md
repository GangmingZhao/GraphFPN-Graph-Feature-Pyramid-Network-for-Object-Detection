# GraphFPN: Graph Feature Pyramid Network for Object Detection with Retinanet
Inspired by Graph_FPN and RetinaNet, we have used the Graph_FPN structure as a backbone to train on Retinanet and the work is not yet complete.   


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

If You need COCO API for test, you can download from [here](https://github.com/cocodataset/cocoapi).
You need to set the backend of DGL to tensorflow, here is tutorial [link](https://docs.dgl.ai/install/index.html#tensorflow-backend)

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
|   ├── data/
|   |    ├── coco
|   |    ├── checkpoint
|   ├── data.zip
|
├── results/
├── src/     
|   ├── configs/
|   |    ├── configs.py
|   |
|   ├── detection/
|   |    ├── datasets/
|   |    |      ├── coco.py
|   |    ├── utils/
|   |
|   ├── model/
|   ├── init_path.py
|   ├── demo.py
|   ├── train.py
|   ├── test.py
├── README.md 
└── requirements.txt
```

## References
[1] Retinanet: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) <br>
[2] Graph-FPN: [GraphFPN: Graph Feature Pyramid Network for Object Detection](https://arxiv.org/abs/2108.00580) <br>
[3] Object Detection with RetinaNet: [Keras Implementation](https://keras.io/examples/vision/retinanet/) <br>
