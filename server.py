import flwr as fl
from flwr.server.strategy import FedAvg

# Define Federated Learning Strategy
strategy = FedAvg(min_available_clients=2, fraction_fit=0.5, fraction_evaluate=0.5)

# Start the Flower server using the new method
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10)  # Fixes the AttributeError
    )

