import asyncio
from typing import List, Optional, Dict, Any, Tuple
import struct
import logging

import numpy as np
import tensorstore as ts
from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TensorStore DVID API",
    description="Python FastAPI implementation of selected DVID endpoints using TensorStore backend",
    version="0.1.0",
)

# Global configuration
class AppConfig:
    ts_specs: Dict[str, Dict[str, Any]] = {}  # Map of dataset name to TensorStore spec

config = AppConfig()

# Models
class DatasetInfo(BaseModel):
    """Information about a dataset"""
    name: str
    dimensions: List[int]
    block_size: List[int]
    voxel_size: List[float]
    voxel_units: List[str]
    data_type: str
    num_channels: int


# Helper functions
async def open_tensorstore(dataset_name: str) -> ts.TensorStore:
    """Open a TensorStore dataset"""
    if dataset_name not in config.ts_specs:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    try:
        dataset = await ts.open(config.ts_specs[dataset_name])
        return dataset
    except Exception as e:
        logger.error(f"Error opening TensorStore: {e}")
        raise HTTPException(status_code=500, detail=f"Error opening TensorStore: {str(e)}")

def block_coords_to_indices(x: int, y: int, z: int, block_size: List[int]) -> Tuple[slice, slice, slice]:
    """Convert block coordinates to data indices"""
    return (
        slice(z * block_size[2], (z + 1) * block_size[2]),
        slice(y * block_size[1], (y + 1) * block_size[1]),
        slice(x * block_size[0], (x + 1) * block_size[0])
    )

def encode_block_data(x: int, y: int, z: int, data: bytes) -> bytes:
    """Encode a block according to DVID specificblocks format"""
    # Format: x(int32), y(int32), z(int32), size(int32), data(bytes)
    header = struct.pack("<iiii", x, y, z, len(data))
    return header + data

async def get_block_data(dataset: ts.TensorStore, x: int, y: int, z: int, block_size: List[int]) -> bytes:
    """Get raw block data from TensorStore"""
    z_slice, y_slice, x_slice = block_coords_to_indices(x, y, z, block_size)
    
    # Get the block data
    block_data = await dataset[z_slice, y_slice, x_slice].read()
    
    # Convert to bytes (assuming uint8 data)
    if block_data.dtype != np.uint8:
        block_data = block_data.astype(np.uint8)
    
    return block_data.tobytes()


# Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TensorStore DVID API", 
        "version": "0.1.0",
        "endpoints": [
            "/api/node/{uuid}/{dataset}/info",
            "/api/node/{uuid}/{dataset}/specificblocks",
            "/api/node/{uuid}/{dataset}/subvolblocks/{size}/{offset}"
        ]
    }

@app.get("/api/node/{uuid}/{dataset}/info")
async def get_info(uuid: str, dataset: str):
    """Get information about a dataset"""
    if dataset not in config.ts_specs:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found")
    
    try:
        ts_store = await open_tensorstore(dataset)
        domain = ts_store.domain
        
        # Extract shape information
        shape = [domain.shape[i] for i in range(domain.rank)]
        
        # Get block size from spec
        block_size = config.ts_specs[dataset].get("chunk_layout", {}).get("chunk_shape", [32, 32, 32])
        
        # Get data type
        data_type = str(ts_store.dtype)
        
        # Default voxel size and units if not specified
        voxel_size = [8.0, 8.0, 8.0]  # Default to 8nm isotropic
        voxel_units = ["nanometers", "nanometers", "nanometers"]
        
        # Return info in DVID format
        return {
            "Base": {
                "TypeName": "uint8blk",
                "DataName": dataset,
                "UUID": uuid,
            },
            "Extended": {
                "BlockSize": block_size,
                "VoxelSize": voxel_size,
                "VoxelUnits": voxel_units
            },
            "Extents": {
                "MinPoint": [0, 0, 0],
                "MaxPoint": shape
            }
        }
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dataset info: {str(e)}")

@app.get("/api/node/{uuid}/{dataset}/specificblocks")
async def get_specific_blocks(
    uuid: str, 
    dataset: str, 
    blocks: str = Query(..., description="Comma-separated list of block coordinates (x,y,z,x,y,z...)"),
    compression: Optional[str] = Query("", description="Compression type (only 'uncompressed' supported)")
):
    """Get specific blocks by coordinates"""
    try:
        # Parse block coordinates
        coords = [int(c) for c in blocks.split(",")]
        if len(coords) % 3 != 0:
            raise HTTPException(status_code=400, detail="Block coordinates must be in groups of 3 (x,y,z)")
        
        ts_store = await open_tensorstore(dataset)
        block_size = config.ts_specs[dataset].get("chunk_layout", {}).get("chunk_shape", [32, 32, 32])
        
        # Create a streaming response
        async def generate_blocks():
            for i in range(0, len(coords), 3):
                x, y, z = coords[i], coords[i+1], coords[i+2]
                try:
                    block_data = await get_block_data(ts_store, x, y, z, block_size)
                    yield encode_block_data(x, y, z, block_data)
                except Exception as e:
                    logger.error(f"Error getting block {x},{y},{z}: {e}")
                    # Skip this block and continue
        
        return StreamingResponse(
            generate_blocks(),
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error in specificblocks: {e}")
        raise HTTPException(status_code=500, detail=f"Error in specificblocks: {str(e)}")

@app.get("/api/node/{uuid}/{dataset}/subvolblocks/{size}/{offset}")
async def get_subvol_blocks(
    uuid: str,
    dataset: str,
    size: str,
    offset: str,
    compression: Optional[str] = Query("", description="Compression type (only 'uncompressed' supported)")
):
    """Get blocks within a subvolume"""
    try:
        # Parse size and offset
        size_parts = [int(s) for s in size.split("_")]
        offset_parts = [int(o) for o in offset.split("_")]
        
        if len(size_parts) != 3 or len(offset_parts) != 3:
            raise HTTPException(status_code=400, detail="Size and offset must have 3 dimensions")
        
        ts_store = await open_tensorstore(dataset)
        block_size = config.ts_specs[dataset].get("chunk_layout", {}).get("chunk_shape", [32, 32, 32])
        
        # Calculate block coordinates
        start_block = [offset_parts[i] // block_size[i] for i in range(3)]
        
        # Calculate how many blocks in each dimension
        blocks_in_dim = [
            (size_parts[i] + block_size[i] - 1) // block_size[i] for i in range(3)
        ]
        
        # Create a streaming response
        async def generate_blocks():
            for z in range(start_block[2], start_block[2] + blocks_in_dim[2]):
                for y in range(start_block[1], start_block[1] + blocks_in_dim[1]):
                    for x in range(start_block[0], start_block[0] + blocks_in_dim[0]):
                        try:
                            block_data = await get_block_data(ts_store, x, y, z, block_size)
                            yield encode_block_data(x, y, z, block_data)
                        except Exception as e:
                            logger.error(f"Error getting block {x},{y},{z}: {e}")
                            # Skip this block and continue
        
        return StreamingResponse(
            generate_blocks(),
            media_type="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Error in subvolblocks: {e}")
        raise HTTPException(status_code=500, detail=f"Error in subvolblocks: {str(e)}")

# Configuration functions
@app.post("/api/config/dataset")
async def configure_dataset(dataset_name: str, ts_spec: Dict[str, Any]):
    """Configure a dataset with a TensorStore spec"""
    config.ts_specs[dataset_name] = ts_spec
    return {"message": f"Dataset {dataset_name} configured successfully"}

@app.get("/api/config/datasets")
async def list_datasets():
    """List all configured datasets"""
    return {"datasets": list(config.ts_specs.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)