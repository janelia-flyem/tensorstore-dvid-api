#!/usr/bin/env python3
"""
Script to create a test dataset using TensorStore.
This creates a simple 3D volume that can be used for testing the API.
"""

import argparse
import asyncio
import numpy as np
import tensorstore as ts
import os

async def create_test_dataset(output_path, size_x=128, size_y=128, size_z=128, block_size=32):
    """Create a test dataset using TensorStore neuroglancer_precomputed driver."""
    
    print(f"Creating test dataset at {output_path} with size {size_x}x{size_y}x{size_z}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create a 3D numpy array with a simple pattern
    data = np.zeros((size_z, size_y, size_x), dtype=np.uint8)
    
    # Create a simple gradient pattern
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                # Simple pattern: value increases with distance from center
                cx, cy, cz = size_x // 2, size_y // 2, size_z // 2
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                # Normalize to 0-255
                data[z, y, x] = min(255, int(dist * 2))
    
    # Create a TensorStore spec for neuroglancer_precomputed
    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {
            "driver": "file",
            "path": output_path,
        },
        "multiscale_metadata": {
            "type": "segmentation",
            "data_type": "uint8",
            "num_channels": 1,
            "scales": [
                {
                    "key": "scale0",
                    "size": [size_x, size_y, size_z],
                    "resolution": [8, 8, 8],
                    "encoding": "raw",
                    "chunk_size": [block_size, block_size, block_size],
                    "voxel_offset": [0, 0, 0],
                }
            ]
        },
        "scale_metadata": {
            "size": [size_x, size_y, size_z],
            "encoding": "raw",
            "chunk_size": [block_size, block_size, block_size],
            "resolution": [8, 8, 8],
            "voxel_offset": [0, 0, 0],
        },
        "scale_index": 0,
        "dtype": "uint8",
        "create": True,
        "delete_existing": True,
    }
    
    # Open the TensorStore for writing
    dataset = await ts.open(spec, create=True, delete_existing=True)
    
    # Write the data
    await dataset.write(data)
    
    print(f"Successfully created test dataset at {output_path}")
    
    # Also print the spec for accessing this dataset (with create=False)
    access_spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {
            "driver": "file",
            "path": output_path,
        },
        "scale_index": 0,
    }
    
    print("\nTo access this dataset, use the following TensorStore spec:")
    print(access_spec)
    
    return access_spec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a test dataset for TensorStore DVID API")
    parser.add_argument("--output", "-o", required=True, help="Output path for the dataset")
    parser.add_argument("--size_x", type=int, default=128, help="Size in X dimension")
    parser.add_argument("--size_y", type=int, default=128, help="Size in Y dimension")
    parser.add_argument("--size_z", type=int, default=128, help="Size in Z dimension")
    parser.add_argument("--block_size", type=int, default=32, help="Block size for chunking")
    
    args = parser.parse_args()
    
    asyncio.run(create_test_dataset(
        args.output, 
        args.size_x, 
        args.size_y, 
        args.size_z, 
        args.block_size
    ))