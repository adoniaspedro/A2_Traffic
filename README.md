1. *PyQt5* - For creating the GUI.
2. *argparse* - For command-line argument parsing.
3. *folium* - For mapping and visualizations.
4. *keras* - For building and training neural network models.
5. *math* - For mathematical functions.
6. *matplotlib* - For plotting and visualizations.
7. *numpy* - For numerical operations.
8. *os* - For interacting with the operating system.
9. *pandas* - For data manipulation and analysis.
10. *sklearn* - For machine learning algorithms.
11. *sys* - For system-specific parameters and functions.
12. *tensorflow* - For machine learning and neural network functionalities.
13. *threading* - For managing threads in your application.
14. *warnings* - For issuing warning messages.
15. *webbrowser* - For opening URLs in a web browser.

Additionally, there are references to local modules (like data, model, and train), which may also be necessary for your project to function correctly. Make sure you have all these libraries installed in your Python environment.

# Traffic Flow Prediction
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).

## Requirement
- Python 3.6    
- Tensorflow-gpu 1.5.0  
- Keras 2.1.3
- scikit-learn 0.19

## Train the model

**Run command below to train the model:**

```
python train.py --model model_name
```

You can choose "lstm", "gru" or "saes" as arguments. The ```.h5``` weight file was saved at model folder.


## Experiment

Data are obtained from the Caltrans Performance Measurement System (PeMS). Data are collected in real-time from individual detectors spanning the freeway system across all major metropolitan areas of the State of California.
	
	device: Tesla K80
	dataset: PeMS 5min-interval traffic flow data
	optimizer: RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
	batch_szie: 256 


**Run command below to run the program:**

```
python main.py
```

These are the details for the traffic flow prediction experiment.


| Metrics | MAE | MSE | RMSE | MAPE |  R2  | Explained variance score |
| ------- |:---:| :--:| :--: | :--: | :--: | :----------------------: |
| LSTM | 7.21 | 98.05 | 9.90 | 16.56% | 0.9396 | 0.9419 |
| GRU | 7.20 | 99.32 | 9.97| 16.78% | 0.9389 | 0.9389|
| SAEs | 7.06 | 92.08 | 9.60 | 17.80% | 0.9433 | 0.9442 |

![evaluate](/images/eva.png)

## Reference

	@article{SAEs,  
	  title={Traffic Flow Prediction With Big Data: A Deep Learning Approach},  
	  author={Y Lv, Y Duan, W Kang, Z Li, FY Wang},
	  journal={IEEE Transactions on Intelligent Transportation Systems, 2015, 16(2):865-873},
	  year={2015}
	}
	
	@article{RNN,  
	  title={Using LSTM and GRU neural network methods for traffic flow prediction},  
	  author={R Fu, Z Zhang, L Li},
	  journal={Chinese Association of Automation, 2017:324-328},
	  year={2017}
	}


## Copyright
See [LICENSE](LICENSE) for details.
