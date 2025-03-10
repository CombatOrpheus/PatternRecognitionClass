import json
from pathlib import Path
from typing import Iterable, Tuple, Literal

from numpy import ndarray

from SPNData import SPNData

LABELS = Literal['network', 'places', 'steady_state', 'mark_density']


class PetriDataLoader:
    """
    Loads Petri Net data from a JSON-L file.
    """

    def __init__(self, source: Path):
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"The specified file '{source}' does not exist or is not a file.")
        self.source = source

    def _load_data(self) -> Iterable[SPNData]:
        """Loads the data and yields PetriNetData instances."""
        with open(self.source, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    yield SPNData(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}. Skipping line.")
                    continue  # Skip to the next line
                except KeyError as e:
                    print(f"Error processing the line: {e}.")
                    continue
                except ValueError as e:
                    print(f"ValueError: {e}.")
                    continue

    def get_spn_data(self) -> Iterable[Tuple[ndarray, ndarray, ndarray]]:
        return (net.to_information() for net in self._load_data())

    def get_labels(self, label: str) -> Iterable:
        data = self._load_data()

        if label == 'network':
            return (net.average_tokens_network for net in data)
        elif label == 'places':
            return (net.average_tokens_per_place for net in data)
        elif label == 'steady_state':
            return (net.steady_state_probabilities for net in data)
        elif label == 'mark_density':
            return (net.token_probability_density_function for net in data)
        else:
            raise ValueError(f"label should be one of {', '.join(LABELS)}.")
