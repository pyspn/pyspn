#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from TorchSPN.src import network, param, nodes
from trainer.trainer import *

print("Loading data set..")
test_raw = genfromtxt('test_mnist_16.csv', delimiter=',')

def segment_data():
    segmented_data = []
    for i in range(10):
        i_examples = test_raw[test_raw[:,0] == i][:,1:] / 255 - 0.5
        segmented_data.append(i_examples)

    return segmented_data

print("Segmenting...")
segmented_data = segment_data()
print("Dataset loaded!")

def compute_prob(model, x):
    (val_dict, cond_mask_dict) = model.network.get_mapped_input_dict(np.array([x]))
    loss = model.network.ComputeTMMLoss(
            val_dict=val_dict,
            cond_mask_dict=cond_mask_dict)
    return loss

def predict(model, x):
    loss = compute_prob(model, x).data.cpu().numpy()

    prediction_index = np.argmin(loss)
    predicted_digit = model.digits[prediction_index]
    return predicted_digit

def main():
    model = pickle.load(open('big250', 'rb'))
    num_tests = 50
    error = 0
    total_data = 0
 
    pdb.set_trace()

    errors = defaultdict(int)
    for i in range(num_tests):
        print("Iteration " + str(i) + ": " + str(error))
        for digit in model.digits:
            x = np.array([ np.tile(segmented_data[digit][i], 40) ] )
            prediction = predict(model, x)
            total_data += 1
            if prediction != digit:
                errors[digit] += 1
                error += 1

    accuracy = 1 - error/total_data
    print("Accuracy: " + str(accuracy * 100) + "%")
    print("Error " + str(error))
    print("Detail " + str(errors))

    pdb.set_trace()

#num_tests = 100
#error = 0
#for test_i in range(num_tests):
#    for digit in range(10):
#        num_tests += 1
#        data = segment_data[digit][test_i]
#        y_pred = model.classify_data(data)
#        if (model.digit == digit) != y_pred:
#            error += 1

if __name__ == '__main__':
    main()
