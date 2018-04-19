import numpy as np

#np.random.rand(1000)

def get_data(fin, idx):
    with open(fin) as f:
        data=[]
        while True:
            line = f.readline()
            if not line:
                break
            one_sample = line.split(',')
            one_sample = [float(x) for x in one_sample]
            data.append(one_sample)


    data = np.array(data)

    data -= np.mean(data, axis=0, keepdims=True)
    data /= np.std(data, axis=0, keepdims=True)
    np.random.shuffle(data)


    data = np.concatenate([ data[:, i].reshape(-1,1) for i in idx ], axis=1)

    num_train = int(len(data)*.6)
    num_cv    = int(len(data)*.8)

    train = data[:num_train]
    val   = data[num_train:num_cv]
    test  = data[num_cv:]

    return train, val, test