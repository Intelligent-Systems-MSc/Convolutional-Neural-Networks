def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    print("In progress, you sould wait at average 7 min for one epoch")
    for e in range(epochs):
        i = 0
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)
            i =i+1
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        
        error /= len(x_train)
        #print(error)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
            
