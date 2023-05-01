# Winning Solution — AutoML Decathlon

This package learns the most predictive mapping of 1-4d inputs to 1-2d targets.<br><br>

### Scope

Coverage is provided for four main areas of machine learning:

- Image Classification:&nbsp; 2-3d -> 1d
- Segmentation:&nbsp; 2-3d -> 2d
- Sequence/Signal:&nbsp; 1-3d-> 1d
- Tabular:&nbsp; 1d -> 1d

All model selection, hyperparameter tuning, and time management occur automatically.<br><br>


### Notes:

All code is optimized for an in-memory dataset, less than 300 MB total package, and ten minutes to ten hours of run-time.

This package won the 2022 AutoML Decathlon hosted by Carnegie Mellon and HP Enterprise (NeurIPS ‘22 Competition), with the best performance on ten out-of-fold tasks.
<br><br>

### Try It:
```
!git clone https://github.com/truefit-ai/auto-ml.git automl
!pip install -r automl/requirements.txt
```

```
from automl.model import Model
from automl.extras import AutoDataset, Metadata
from automl.metrics import *

import torch
import sklearn.datasets as skdatasets
data = skdatasets.load_wine()
```

```
x_data, y_data = data['data'], data['target']
train_data, test_data = [list(zip(x_data[i::2], y_data[i::2])) for i in range(2)]

def convert(data):
    return AutoDataset([(torch.tensor(x).unsqueeze(0), 
        torch.nn.functional.one_hot(torch.tensor(y), y_data.max() + 1))
                             for x, y in data], 
                Metadata('single-label', 'zero_one_error'))
train_dataset = convert(train_data)
test_dataset = convert(test_data)
```

```
model = Model(train_dataset.metadata)
model.train(train_dataset, remaining_time_budget = 0.05 * 3600)
yp = model.test(test_dataset)
print('Accuracy: {:.1%}'.format(1 - zero_one_error(yp, 
            torch.stack([e[1] for e in test_dataset]))))
```
<br>

### License

All rights reserved; incorporation into externally released or promoted automated machine learning packages is prohibited.

You are granted a perpetual, royalty-free license to use this content for any other purpose, including internal model training and deployment, as well as external uses that are not intended for automated machine learning systems.

If more than twelve months elapse without an update to this repository, this license automatically converts to an MIT License in all cases.
