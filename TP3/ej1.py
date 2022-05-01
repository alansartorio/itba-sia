from activation_functions import step_func
from network import Network
from train_data import ej1_and_data, ej1_xor_data

# model = Network.with_zeroed_weights(2, (1, ), step_func)
model = Network.with_random_weights(2, (1, ), step_func)
data = ej1_and_data
# data = ej1_xor_data

# print(model.layers[0].weights.flatten())
# print(model.error(ej1_data))
for i in range(10):
    model.train(0.1, data)
    print(model.layers[0].weights.flatten(), model.error(data))
    # print(model.layers[0].weights.flatten(), model.error(ej1_data))
    # for single_data in data:
        # print(single_data.inputs, single_data.outputs, model.evaluate(single_data.inputs), end=' | ')
    # print()
# single_neuron = Network.with_random_weights(1, (2, 3), step_func)


# print(single_neuron.evaluate(np.array([1])))
