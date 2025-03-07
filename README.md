# TensorStore DVID API

A Python-based FastAPI service that provides HTTP API support for selected DVID endpoints, using TensorStore as the backend data source.

## Implemented Endpoints

This service implements the following DVID endpoints from the original DVID server:

1. `/api/node/{uuid}/{dataset}/info` - Get information about a dataset
2. `/api/node/{uuid}/{dataset}/specificblocks` - Get specific blocks by coordinates
3. `/api/node/{uuid}/{dataset}/subvolblocks/{size}/{offset}` - Get blocks within a subvolume

These endpoints are compatible with the DVID API format but use TensorStore as the backend data source instead of the original storage drivers in DVID.

## Implementation Details

- **FastAPI**: We use FastAPI for efficient and concurrent HTTP API handling
- **TensorStore Backend**: All data operations are performed using Google's TensorStore library
- **Streaming Responses**: For large data requests, responses are streamed to avoid memory issues
- **Dynamic Configuration**: Datasets can be configured at runtime through API endpoints
- **Asynchronous Processing**: Uses Python's async/await for efficient I/O operations

## Setup

### Prerequisites

- Python 3.8+
- TensorStore Python library
- FastAPI and dependencies

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/tensorstore-dvid-api.git
cd tensorstore-dvid-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Service

Start the service:

```bash
cd src
python main.py
```

The API will be available at `http://localhost:8000`. You can access the automatic API documentation at `http://localhost:8000/docs`.

## Docker Deployment

You can also deploy the service using Docker:

```bash
docker build -t tensorstore-dvid-api .
docker run -p 8000:8000 tensorstore-dvid-api
```

## Configuring Datasets

Before using the API, you need to configure at least one dataset by providing a TensorStore spec. This can be done via the `/api/config/dataset` endpoint:

```bash
curl -X POST "http://localhost:8000/api/config/dataset?dataset_name=my_dataset" \
     -H "Content-Type: application/json" \
     -d '{
           "driver": "neuroglancer_precomputed",
           "kvstore": {
             "driver": "file",
             "path": "/path/to/dataset"
           }
         }'
```

You can list all configured datasets using:

```bash
curl -X GET "http://localhost:8000/api/config/datasets"
```

## Example Usage

### Get dataset info

```bash
curl -X GET "http://localhost:8000/api/node/dummy_uuid/my_dataset/info"
```

### Get specific blocks

```bash
curl -X GET "http://localhost:8000/api/node/dummy_uuid/my_dataset/specificblocks?blocks=0,0,0,1,0,0"
```

### Get subvolume blocks

```bash
curl -X GET "http://localhost:8000/api/node/dummy_uuid/my_dataset/subvolblocks/64_64_64/0_0_0"
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

## License

BSD 3-Clause (See `LICENSE` file)
