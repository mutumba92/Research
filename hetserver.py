import flwr as fl
from heterogeneousCNN import FedAsyncCustom  # Import custom strategy

# Define federated learning strategy
strategy = FedAsyncCustom(initial_parameters=None)

if __name__ == "__main__":
    # Start the server using the new strategy
    fl.server.start_server(
        server_address="0.0.0.0:9090", 
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10)  # Specifies the number of federated rounds
    )