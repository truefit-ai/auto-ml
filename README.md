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

There are better vision architectures, better augmentation strategies, more unified factoring, and methods more suited to either very large or very small datasets--just not in this package. <br><br>


### Try It:
```
!pip install -r automl/requirements.txt
!git clone https://github.com/truefit-ai/auto-ml.git
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
    return AutoDataset([(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0), 
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
print('Accuracy: {:.1%}'.format(1 - zero_one_error(yp, torch.stack([e[1] for e in test_dataset]))))
```
<br>

### License
All rights reserved; may not be incorporated into automated machine learning packages.

You are granted a perpetual and royalty-free license to use for any other purpose, including model training and deployment in any form that is not intended as an automated machine system.

This license automatically converts to an MIT License if more than twelve months elapse without update to this repo.

