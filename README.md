Running
path_to_compiled --algorithm "RF" --trainInput "file:///Users/doruchiulan/Desktop/training_data.csv" --testInput "file:///Users/doruchiulan/Desktop/test_data.csv"  --algoNumTrees 5 --algoMaxDepth 10 --algoMaxBins 2868  --numFolds 3 --trainSample 1.0 --testSample 1.0

path_to_compiled --algorithm "GBT" --trainInput "file:///Users/doruchiulan/Desktop/training_data.csv" --testInput "file:///Users/doruchiulan/Desktop/test_data.csv"  --algoMaxIter 30 --algoMaxDepth 10 --algoMaxBins 2868  --numFolds 10 --trainSample 1.0 --testSample 1.0

Example output (Random Forest 5 Trees, 10 Depth)
Label is actual value in --testInput, prediction is predicted value
+-----+------------------+
|label|        prediction|
+-----+------------------+
|  107| 92.19870107057952|
|   74| 66.74340510002506|
|   86| 87.37438388732222|
|  152|114.59119100891398|
|   85| 67.37065916533724|
|   78| 81.75919161112247|
|   36| 40.35997554255768|
|   94| 65.98954238542692|
|  165|161.38520014201825|
|   75| 89.86796637978934|
|   16|16.581978546627674|
|   65| 64.09520239906604|
|    2|1.4314465808587928|
|  120|138.73884957719616|
|   53| 97.33208923295703|
|   48| 64.34586433862243|
|  152|140.70023416219388|
|    1|1.4314465808587928|
|   18| 9.618659482951902|
|   35| 64.65829844342657|
+-----+------------------+